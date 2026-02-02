#!/usr/bin/env python3
"""
DLL Python 래퍼 코드 생성기
C++ 헤더 파일을 분석하여 ctypes 기반 Python 래퍼 클래스를 생성합니다.
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple

TYPE_MAP = {
    'int': ('ctypes.c_int', 'int'),
    'int32_t': ('ctypes.c_int32', 'int'),
    'int64_t': ('ctypes.c_int64', 'int'),
    'uint32_t': ('ctypes.c_uint32', 'int'),
    'uint64_t': ('ctypes.c_uint64', 'int'),
    'size_t': ('ctypes.c_size_t', 'int'),
    'float': ('ctypes.c_float', 'float'),
    'double': ('ctypes.c_double', 'float'),
    'bool': ('ctypes.c_bool', 'bool'),
    'char*': ('ctypes.c_char_p', 'bytes'),
    'const char*': ('ctypes.c_char_p', 'str'),
    'void': ('None', 'None'),
}

def parse_function(line: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    """함수 선언을 파싱하여 (이름, 반환타입, [(파라미터타입, 파라미터명)]) 반환"""
    # API int function_name(int a, double b);
    pattern = r'API\s+(\w+\*?)\s+(\w+)\s*\((.*?)\)'
    match = re.search(pattern, line)
    if not match:
        return None

    ret_type = match.group(1)
    func_name = match.group(2)
    params_str = match.group(3).strip()

    params = []
    if params_str and params_str != 'void':
        for param in params_str.split(','):
            param = param.strip()
            # const char* name 또는 int name
            parts = param.rsplit(' ', 1)
            if len(parts) == 2:
                params.append((parts[0].strip(), parts[1].strip()))

    return func_name, ret_type, params

def generate_wrapper(header_path: str, dll_name: str) -> str:
    """헤더 파일로부터 Python 래퍼 코드 생성"""
    header = Path(header_path).read_text()

    functions = []
    for line in header.split('\n'):
        if 'API' in line and '(' in line:
            parsed = parse_function(line)
            if parsed:
                functions.append(parsed)

    class_name = dll_name.replace('-', '_').title().replace('_', '')

    code = f'''#!/usr/bin/env python3
"""
Auto-generated ctypes wrapper for {dll_name}.dll
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional

class {class_name}:
    def __init__(self, dll_path: Optional[str] = None):
        if dll_path is None:
            dll_path = Path(__file__).parent / "{dll_name}.dll"
        self._dll = ctypes.CDLL(str(dll_path))
        self._setup_functions()

    def _setup_functions(self):
'''

    for func_name, ret_type, params in functions:
        ctype_ret = TYPE_MAP.get(ret_type, ('ctypes.c_void_p', 'any'))[0]
        ctype_args = [TYPE_MAP.get(p[0], ('ctypes.c_void_p', 'any'))[0] for p in params]

        code += f'        # {func_name}\n'
        code += f'        self._dll.{func_name}.argtypes = [{", ".join(ctype_args)}]\n'
        code += f'        self._dll.{func_name}.restype = {ctype_ret}\n\n'

    for func_name, ret_type, params in functions:
        py_ret = TYPE_MAP.get(ret_type, ('any', 'any'))[1]
        py_params = [f'{p[1]}: {TYPE_MAP.get(p[0], ("any", "any"))[1]}' for p in params]
        param_names = [p[1] for p in params]

        code += f'''
    def {func_name}(self, {", ".join(py_params)}) -> {py_ret}:
        return self._dll.{func_name}({", ".join(param_names)})
'''

    return code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Python ctypes wrapper from C++ header')
    parser.add_argument('header', help='Path to C++ header file')
    parser.add_argument('--name', '-n', default='mydll', help='DLL name (without .dll)')
    parser.add_argument('--output', '-o', help='Output Python file path')

    args = parser.parse_args()

    wrapper_code = generate_wrapper(args.header, args.name)

    if args.output:
        Path(args.output).write_text(wrapper_code)
        print(f"Generated: {args.output}")
    else:
        print(wrapper_code)
