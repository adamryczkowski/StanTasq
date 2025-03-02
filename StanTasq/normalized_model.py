from pydantic import BaseModel
from .model import StanModel
from ._cmdstan_runner import initialize
from .utils import normalize_stan_model_by_file, normalize_stan_model_by_str
from pathlib import Path
from typing import Optional
from entityhash import calc_hash, EntityHash


# noinspection PyDataclass
class NormalizedModel(BaseModel):
    """
    Class that holds the Stan model code in its normalized form.
    Its instantiation requires cmdstan, and is expected to be instantiated only on workers.
    """

    model_path: Optional[Path]
    model_name: str
    stanc_opts: dict[str, str] = {}

    messages: dict[str, str] = {}

    @staticmethod
    def FromStanModel(model: StanModel) -> "NormalizedModel":
        initialize()

        if isinstance(model.source_code, Path):
            model_filename_tmp, msg = normalize_stan_model_by_file(model.source_code)
        else:
            model_filename_tmp, msg = normalize_stan_model_by_str(model.source_code)

        return NormalizedModel(
            model_path=model_filename_tmp,
            model_name=model.model_name,
            stanc_opts=model.stanc_opts,
            messages=msg,
        )

    @property
    def model_str(self):
        return self.model_path.read_text()

    @property
    def model_hash(self) -> EntityHash:
        return calc_hash([self.model_str, self.stanc_opts])
