from dataclasses import dataclass
from src.models.balance import Balance


@dataclass
class User:
    is_admin: bool
    username: str
    email: str
    balance: Balance
    password_hash: str
