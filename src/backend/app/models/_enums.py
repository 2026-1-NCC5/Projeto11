from enum import StrEnum


class UserRole(StrEnum):
    professor = "professor"
    aluno = "aluno"


class PeriodType(StrEnum):
    matutino = "matutino"
    noturno = "noturno"


class FoodCategory(StrEnum):
    arroz = "arroz"
    feijao = "feijao"
    acucar = "acucar"
    macarrao = "macarrao"
    oleo = "oleo"
    fuba = "fuba"
