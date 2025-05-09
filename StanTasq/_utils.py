# import subprocess
# import io
# import json
# import tempfile
# from contextlib import redirect_stdout, redirect_stderr
# from hashlib import sha256
# from pathlib import Path
# import pickle
# from typing import Any, Optional
#
# import cmdstanpy
# import numpy as np
#
# from .ifaces import StanOutputScope
#
# # from EntityHash import calc_inthash, make_dict_serializable_in_place
# import base64
#
#
# def find_model_in_cache(
#     model_cache: Path, model_name: str, model_hash: int | str
# ) -> Path:
#     best_model_filename = None
#     if isinstance(model_hash, int):
#         model_id = base64.b64encode(
#             abs(model_hash).to_bytes(32, byteorder="big")
#         ).decode("utf-8")
#     else:
#         model_id = model_hash
#     for hash_char_count in range(0, len(model_id)):
#         if hash_char_count == 0:
#             model_filename = model_cache / f"{model_name}.stan"
#         else:
#             model_filename = (
#                 model_cache / f"{model_name} {model_id[:hash_char_count]}.stan"
#             )
#         if model_filename.exists():
#             # load the model and check its hash if it matches
#             with model_filename.open("r") as f:
#                 model_code = f.read()
#             if (
#                 base64.b64encode(
#                     abs(calc_inthash(model_code.encode())).to_bytes(32, byteorder="big")
#                 ).decode("utf-8")
#                 == model_id
#             ):
#                 if hash_char_count == len(model_id) - 1:
#                     raise RuntimeError(
#                         f"Hash collision for model {model_name} with hash {model_id} in cache file {model_filename}!"
#                     )
#                 return model_filename
#         else:
#             if best_model_filename is None:
#                 best_model_filename = model_filename
#
#         hash_char_count += 1
#
#     assert best_model_filename is not None
#     return best_model_filename
#
#
# def get_compiled_model_hashes(model_cache: Path) -> dict[str, Path]:
#     ans = {}
#     for model_file in model_cache.glob("*.stan"):
#         with model_file.open("r") as f:
#             model_code = f.read()
#         model_hash = sha256(model_code.encode()).hexdigest()
#         ans[model_file.stem] = model_hash
#     return ans
#
#
# def denumpy_dict(d: dict[str, Any]) -> dict[str, Any]:
#     ans = {}
#     for key in d:
#         if isinstance(d[key], np.ndarray):
#             ans[key] = d[key].tolist()
#         elif isinstance(d[key], dict):
#             ans[key] = denumpy_dict(d[key])
#         else:
#             ans[key] = d[key]
#     return ans
#
#
# def model2json(
#     stan_code: str, model_name: str, data: dict, output_type: StanOutputScope, **kwargs
# ) -> str:
#     assert isinstance(stan_code, str)
#     assert isinstance(model_name, str)
#     assert isinstance(data, dict)
#     # assert isinstance(engine, StanResultEngine)
#     assert isinstance(output_type, StanOutputScope)
#
#     out = {}
#     out["model_code"] = stan_code
#     out["model_name"] = model_name
#     out["data"] = denumpy_dict(data)
#     out["output_type"] = output_type.txt_value()
#     out.update(kwargs)
#
#     # Convert out to json
#     return str(json.dumps(out))
#
#
# def normalize_stan_model_by_str(stan_code: str) -> tuple[Optional[str], dict[str, str]]:
#     # Write the model to a disposable temporary location
#     with tempfile.NamedTemporaryFile("w", delete=True) as f:
#         f.write(stan_code)
#         f.flush()
#         model_filename = Path(f.name)
#
#         file_out, msg = normalize_stan_model_by_file(model_filename)
#         if file_out is None:
#             return None, msg
#
#         return file_out.read_text(), msg
#
#
# def normalize_stan_model_by_file(
#     stan_file: str | Path,
# ) -> tuple[Optional[Path], dict[str, str]]:
#     if isinstance(stan_file, str):
#         stan_file = Path(stan_file)
#     assert isinstance(stan_file, Path)
#     assert stan_file.exists()
#     assert stan_file.is_file()
#
#     stdout = io.StringIO()
#     stderr = io.StringIO()
#
#     # Copy stan_file to temporary location
#     tmpfile = tempfile.NamedTemporaryFile("w", delete=False)
#     tmpfile.write(stan_file.read_bytes().decode())
#     tmpfile.close()
#
#     with redirect_stdout(stdout), redirect_stderr(stderr):
#         try:
#             cmdstanpy.format_stan_file(
#                 tmpfile.name,
#                 overwrite_file=True,
#                 canonicalize=[
#                     "deprecations",
#                     "parentheses",
#                     "braces",
#                     "includes",
#                     "strip-comments",
#                 ],
#             )
#         except subprocess.CalledProcessError as e:
#             msg = {
#                 "stanc_output": stdout.getvalue() + e.stdout,
#                 "stanc_warning": stderr.getvalue(),
#                 "stanc_error": e.stderr,
#             }
#             return None, msg
#
#     msg = {"stanc_output": stdout.getvalue(), "stanc_warning": stderr.getvalue()}
#
#     return Path(tmpfile.name), msg
#
#
# def serialize_to_bytes(obj: Any, format: str) -> bytes:
#     if format == "pickle":
#         if isinstance(obj, dict):
#             make_dict_serializable_in_place(obj)
#         return pickle.dumps(obj)
#     else:
#         raise Exception(f"Unknown format {format}")
