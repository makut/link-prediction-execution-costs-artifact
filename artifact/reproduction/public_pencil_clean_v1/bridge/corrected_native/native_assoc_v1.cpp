// Corrected-native control v1: persistent association map / touched-only reset.
// Scan order and outputs are adapted from pytorch_sparse 0.6.18 (MIT),
// commit 7d22892fb47626d2c5b0171769d44bdc8edd606c,
// csrc/cpu/saint_cpu.cpp blob 21a13be8067e66ef4d6f9ae57f98407332f31929.
// Exact upstream license is retained in third_party/pytorch_sparse_LICENSE.
// This is a separately labeled standalone control, NOT an unmodified Torch op.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

struct NativeResultV1 {
    std::int64_t* rows = nullptr;
    std::int64_t* cols = nullptr;
    std::int64_t* edge_ids = nullptr;
    std::int64_t length = 0;
};

struct NativeStateV1 {
    std::int64_t n;
    int mode; // 0 = fresh full-N clear per call; 1 = persistent touched reset.
    std::vector<std::int64_t> persistent_assoc;
    std::int64_t live_touched = 0;
    std::mutex mutex;
    NativeStateV1(std::int64_t n_, int mode_) : n(n_), mode(mode_) {
        if (mode == 1) persistent_assoc.assign(static_cast<std::size_t>(n), -1);
    }
};

static void report_error(char* out, std::size_t cap, const char* value) noexcept {
    if (out && cap) std::snprintf(out, cap, "%s", value);
}

// Construct BEFORE changing assoc. Every touched slot is restored, including
// exceptional exits during the scan, output allocation, or deliberate fixtures.
class TouchedResetV1 {
    NativeStateV1& state;
    std::int64_t* assoc;
    const std::int64_t* idx;
    std::int64_t count = 0;
public:
    TouchedResetV1(NativeStateV1& state_, std::int64_t* assoc_, const std::int64_t* idx_)
        : state(state_), assoc(assoc_), idx(idx_) {}
    TouchedResetV1(const TouchedResetV1&) = delete;
    TouchedResetV1& operator=(const TouchedResetV1&) = delete;
    void note_touch() noexcept { ++count; state.live_touched = count; }
    ~TouchedResetV1() noexcept {
        if (state.mode == 1) {
            for (std::int64_t i = 0; i < count; ++i) assoc[idx[i]] = -1;
        }
        state.live_touched = 0;
    }
};

static std::unique_ptr<std::int64_t[]> cloned(const std::vector<std::int64_t>& v) {
    std::unique_ptr<std::int64_t[]> out;
    if (!v.empty()) {
        out.reset(new std::int64_t[v.size()]);
        std::memcpy(out.get(), v.data(), v.size() * sizeof(std::int64_t));
    }
    return out;
}

extern "C" {
const char* lpp_native_version_v1() { return "persistent_assoc_touched_reset_v1"; }

int lpp_native_create_v1(std::int64_t n, int mode, std::int64_t max_assoc_bytes,
                         NativeStateV1** result, char* error, std::size_t error_cap) noexcept {
    if (result) *result = nullptr;
    try {
        if (!result || n <= 0 || (mode != 0 && mode != 1))
            throw std::invalid_argument("invalid state dimensions/mode");
        if (max_assoc_bytes <= 0 || n > std::numeric_limits<std::int64_t>::max() / 8 ||
            n > max_assoc_bytes / 8 || static_cast<std::uint64_t>(n) >
            std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t))
            throw std::invalid_argument("association allocation exceeds explicit byte budget");
        *result = new NativeStateV1(n, mode);
        return 0;
    } catch (const std::exception& ex) { report_error(error, error_cap, ex.what()); return 1; }
      catch (...) { report_error(error, error_cap, "unknown state initialization failure"); return 2; }
}

void lpp_native_destroy_v1(NativeStateV1* state) noexcept { delete state; }
std::int64_t lpp_native_persistent_bytes_v1(const NativeStateV1* state) noexcept {
    return state && state->mode == 1 ? 8 * state->n : 0;
}
std::int64_t lpp_native_live_touched_v1(const NativeStateV1* state) noexcept {
    return state ? state->live_touched : -1;
}
void lpp_native_result_free_v1(NativeResultV1* out) noexcept {
    if (!out) return;
    delete[] out->rows; delete[] out->cols; delete[] out->edge_ids;
    out->rows = out->cols = out->edge_ids = nullptr; out->length = 0;
}

int lpp_native_induce_v1(NativeStateV1* state,
        const std::int64_t* rowptr, std::int64_t ptr_len,
        const std::int64_t* col, std::int64_t m,
        const std::int64_t* idx, std::int64_t k,
        std::int64_t fixture_fail_after_touches,
        NativeResultV1* out, char* error, std::size_t error_cap) noexcept {
    // Caller owns an initially empty result. No C++ exception crosses the ABI.
    if (!out) return 1;
    out->rows = out->cols = out->edge_ids = nullptr; out->length = 0;
    try {
        if (!state || !rowptr || !idx || ptr_len != state->n + 1 || m < 0 ||
            (m > 0 && !col) || k <= 0 || k > state->n)
            throw std::invalid_argument("invalid CSR or selected-set dimensions");
        std::unique_lock<std::mutex> lock(state->mutex, std::try_to_lock);
        if (!lock.owns_lock()) throw std::runtime_error("state is per-worker; concurrent calls forbidden");
        if (state->live_touched != 0) throw std::runtime_error("dirty association state before call");
        if (rowptr[0] != 0 || rowptr[state->n] != m)
            throw std::invalid_argument("invalid CSR first/terminal offset");
        // O(k) bounds and O(k log k) duplicate safety checks are identical in
        // both standalone arms. They are extra to upstream's assumed contract.
        std::vector<std::int64_t> unique_idx(idx, idx + k);
        std::sort(unique_idx.begin(), unique_idx.end());
        for (std::int64_t i = 0; i < k; ++i) {
            if (unique_idx[i] < 0 || unique_idx[i] >= state->n ||
                (i && unique_idx[i] == unique_idx[i - 1]))
                throw std::invalid_argument("selected IDs must be distinct and in range");
        }
        // Validate ALL IDs before any original-order rowptr dereference: an
        // out-of-range first idx may sort behind valid lower IDs above.
        for (std::int64_t i = 0; i < k; ++i) {
            const auto a = rowptr[idx[i]], b = rowptr[idx[i] + 1];
            if (a < 0 || b < a || b > m) throw std::invalid_argument("invalid selected CSR row");
        }
        std::vector<std::int64_t> full_assoc;
        if (state->mode == 0) full_assoc.assign(static_cast<std::size_t>(state->n), -1);
        std::int64_t* assoc = state->mode == 1 ? state->persistent_assoc.data() : full_assoc.data();
        TouchedResetV1 reset(*state, assoc, idx);
        for (std::int64_t i = 0; i < k; ++i) {
            assoc[idx[i]] = i;
            reset.note_touch();
            if (fixture_fail_after_touches >= 0 && i + 1 == fixture_fail_after_touches)
                throw std::runtime_error("injected fixture failure after touches");
        }
        // Same selected-row scan and append order as pinned saint_cpu.cpp.
        // Fresh output vectors and three fresh cloned outputs in BOTH arms.
        std::vector<std::int64_t> rows, cols, edge_ids;
        for (std::int64_t v_new = 0; v_new < k; ++v_new) {
            const auto v = idx[v_new];
            for (std::int64_t j = rowptr[v]; j < rowptr[v + 1]; ++j) {
                const auto w = col[j];
                if (w < 0 || w >= state->n) throw std::invalid_argument("neighbor outside graph");
                const auto w_new = assoc[w];
                if (w_new > -1) {
                    rows.push_back(v_new); cols.push_back(w_new); edge_ids.push_back(j);
                }
            }
        }
        auto result_rows = cloned(rows), result_cols = cloned(cols), result_ids = cloned(edge_ids);
        out->length = static_cast<std::int64_t>(rows.size());
        out->rows = result_rows.release(); out->cols = result_cols.release(); out->edge_ids = result_ids.release();
        // reset's destructor clears only idx slots before the call returns.
        return 0;
    } catch (const std::exception& ex) { report_error(error, error_cap, ex.what()); return 1; }
      catch (...) { report_error(error, error_cap, "unknown native induction failure"); return 2; }
}
} // extern C
