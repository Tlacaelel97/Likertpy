"""
List with differtent type of scales that could be used in 
a Likert Scale
"""
import typing

Scale = typing.List[str]

msas_G1: Scale = [
    "No me pasó",
    "Casi nunca",
    "Algunas veces",
    "Frecuentemente",
    "Casi todo el tiempo",
]

msas_G2: Scale = [
    "No me pasó",
    "Leve",
    "Regular",
    "Grave",
    "Muy grave",
]

msas_G3: Scale = [
    "Nada",
    "Un poco",
    "Regular",
    "Bastante",
    "Mucho",
]

msas_G4: Scale = [
    "No lo tiene_1",
    "Leve",
    "Regular_1",
    "Grave",
    "Muy grave",
    "No lo tiene_2",
    "Nada",
    "Un poco",
    "Regular_2",
    "Bastante",
    "Mucho",
]

apca: Scale = [
    0,1,2,3,4,5
]

pedsql: Scale = [
    "Nunca",
    "Casi Nunca",
    "Algunas Veces",
    "A Menudo",
    "Casi Siempre",
]

pedsql_2: Scale = [
    "Nunca",
    "Casi Nunca",
    "Algunas Veces",
    "A Menudo",
    "Muchas Veces",
]

pedsql_5: Scale = [
    "Nunca",
    "Casi nunca",
    "Algunas veces",
    "Con frecuencia",
    "Casi siempre",
]

agree5_0: Scale = [
    "0",
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]

agree5: Scale = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]
agree: Scale = agree5

acceptable5_0: Scale = [
    "0",
    "Completely unacceptable",
    "Somewhat unacceptable",
    "Neutral",
    "Somewhat acceptable",
    "Completely acceptable",
]

acceptable5: Scale = [
    "Completely unacceptable",
    "Somewhat unacceptable",
    "Neutral",
    "Somewhat acceptable",
    "Completely acceptable",
]
acceptable: Scale = acceptable5

likely5: Scale = [
    "Very unlikely",
    "Somewhat unlikely",
    "Neutral",
    "Somewhat likely",
    "Very likely",
]
likely: Scale = likely5

scores5_0: Scale = [
    "0",
    "1 - Strongly Disagree",
    "2 - Disagree",
    "3 - Neither Agree nor Disagree",
    "4 - Agree",
    "5 - Strongly Agree",
]

scores5: Scale = [
    "1 - Strongly Disagree",
    "2 - Disagree",
    "3 - Neither Agree nor Disagree",
    "4 - Agree",
    "5 - Strongly Agree",
]

scores6_0: Scale = [
    "0",
    "1 - Strongly Disagree",
    "2 - Disagree",
    "3 - Slightly Disagree",
    "4 - Slightly Agree",
    "5 - Agree",
    "6 - Strongly Agree",
]

scores6: Scale = [
    "1 - Strongly Disagree",
    "2 - Disagree",
    "3 - Slightly Disagree",
    "4 - Slightly Agree",
    "5 - Agree",
    "6 - Strongly Agree",
]

scores7_0: Scale = [
    "0",
    "1 - Strongly Disagree",
    "2 - Disagree",
    "3 - Slightly Disagree",
    "4 - Neither Agree nor Disagree",
    "5 - Slightly Agree",
    "6 - Agree",
    "7 - Strongly Agree",
]

scores7: Scale = [
    "1 - Strongly Disagree",
    "2 - Disagree",
    "3 - Slightly Disagree",
    "4 - Neither Agree nor Disagree",
    "5 - Slightly Agree",
    "6 - Agree",
    "7 - Strongly Agree",
]

raw5_0: Scale = ["0", "1", "2", "3", "4", "5"]
raw5: Scale = ["1", "2", "3", "4", "5"]

raw6_0: Scale = ["0", "1", "2", "3", "4", "5", "6"]

raw6: Scale = ["1", "2", "3", "4", "5", "6"]

raw7_0: Scale = ["0", "1", "2", "3", "4", "5", "6", "7"]

raw7: Scale = ["1", "2", "3", "4", "5", "6", "7"]