"""Python3.10/3.12 AST identity: remove only empty PEP695 type_params fields.

No statement, expression, argument, annotation, decorator, default, return,
type comment or other AST field is removed. Generic type parameters are refused.
Source bytes are independently pinned as well as the portable AST identities.
"""
import ast
import copy
import hashlib
import sys

VERSION = 'py310_ast_identity_only_empty_pep695_type_params_v1'
ALLOWED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def canonical_dump(node):
    node = copy.deepcopy(node)
    for value in ast.walk(node):
        if 'type_params' in value._fields:
            if not isinstance(value, ALLOWED):
                raise ValueError('type_params is unsupported on this AST node')
            if not hasattr(value, 'type_params') or not isinstance(value.type_params, list) or value.type_params:
                raise ValueError('only explicitly empty type_params may be normalized')
            value._fields = tuple(name for name in value._fields if name != 'type_params')
    options = {'annotate_fields': True, 'include_attributes': False}
    # Python3.13 changed the display default for empty fields. Force their
    # original display; this preserves fields rather than removing any others.
    if sys.version_info >= (3, 13): options['show_empty'] = True
    return ast.dump(node, **options)


def canonical_sha256(node):
    return hashlib.sha256(canonical_dump(node).encode()).hexdigest()
