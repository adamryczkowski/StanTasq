from pydantic import BaseModel
from pathlib import Path
import prettytable


# noinspection PyDataclass
class StanModel(BaseModel):
    """
    Class that holds the Stan model code. This is a Pydantic model.
    """

    source_code: str | Path
    model_name: str
    stanc_opts: dict[str, str] = {}

    def __repr__(self) -> str:
        if self.model_name != "":
            ans = f"Stan model {self.model_name}"
        else:
            ans = "Stan model"
        if isinstance(self.source_code, Path):
            ans = f" stored as {self.source_code.name}"

        if len(self.stanc_opts) > 0:
            if len(self.stanc_opts) == 1:
                opt_name, opt_value = next(iter(self.stanc_opts.items()))

                ans += f" with stanc option {opt_name}={opt_value}"
            else:
                table = prettytable.PrettyTable()
                ans += " with the following stanc options:"
                table.field_names = ["option name", "value"]
                for opt_name, opt_value in self.stanc_opts.items():
                    table.add_row([opt_name, opt_value])

                ans += str(table)

        return ans
