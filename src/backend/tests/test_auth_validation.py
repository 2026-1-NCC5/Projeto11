"""Validação de schemas Pydantic — não toca o banco."""
import pytest
from app.models._enums import PeriodType
from app.schemas.auth import RegisterAlunoIn, RegisterIn, RegisterProfessorIn
from pydantic import TypeAdapter, ValidationError

register_adapter = TypeAdapter(RegisterIn)


def _prof_payload(**over):
    base = {
        "role": "professor",
        "email": "alguem@fecap.br",
        "ra": "123456",
        "full_name": "Prof Teste",
        "password": "Senha@123",
        "period": "matutino",
    }
    base.update(over)
    return base


def _aluno_payload(**over):
    base = {
        "role": "aluno",
        "email": "alguem@edu.fecap.br",
        "ra": "24026298",
        "full_name": "Aluno Teste",
        "password": "Senha@123",
        "period": "matutino",
        "course": "Administração",
        "semester": 1,
    }
    base.update(over)
    return base


def test_professor_ok():
    obj = register_adapter.validate_python(_prof_payload())
    assert isinstance(obj, RegisterProfessorIn)
    assert obj.email == "alguem@fecap.br"
    assert obj.period == PeriodType.matutino


def test_aluno_ok():
    obj = register_adapter.validate_python(_aluno_payload())
    assert isinstance(obj, RegisterAlunoIn)
    assert obj.semester == 1
    assert obj.course == "Administração"


@pytest.mark.parametrize(
    "field,value",
    [
        ("email", "alguem@gmail.com"),
        ("email", "alguem@edu.fecap.br"),
        ("ra", "12345"),
        ("ra", "1234567"),
        ("password", "fraca1!"),
        ("password", "SemSimbolo1"),
    ],
)
def test_professor_invalid(field, value):
    with pytest.raises(ValidationError):
        register_adapter.validate_python(_prof_payload(**{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("email", "alguem@fecap.br"),
        ("ra", "1234567"),
        ("ra", "123456789"),
        ("course", "Engenharia"),
        ("semester", 0),
        ("semester", 9),
        ("password", "abcdefgh"),
    ],
)
def test_aluno_invalid(field, value):
    with pytest.raises(ValidationError):
        register_adapter.validate_python(_aluno_payload(**{field: value}))
