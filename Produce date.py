import csv
import random


class GenerateDate:
    separate_lst = ["/", ",", "-", " ", "'"]
    def __init__(self):

    # dd,mm,yyyy
    def type_1(self) -> list:
        lst_rtn = []
        for i in range(50):
            date = str(random.randint(1, 31))
            month = str(random.randint(1, 12))
            year = str(random.randint(1900, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 5)
            string_date = date + self.separate_lst[separate1] + month + self.separate_lst[separate2] + year
            result = ""
            for j in range(len(date)): result += "d"
            result += "s"
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(year)): result += "y"

        for i in range(50):
            date = random.randint(1, 31)
            if date < 10:
                date = "0" + str(date)
            else:
                date = str(date)
            month = str(random.randint(1, 12))
            year = str(random.randint(1900, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 5)
            string_date = date + self.separate_lst[separate1] + month + self.separate_lst[separate2] + year
            result = ""
            result += "dd"
            result += "s"
            result += "mm"
            result += "s"
            result += "yyyy"


        return

    # mm,dd,yyyy
    def type_2(self) -> list:
        lst_rtn = []
        for i in range(50):
            date = str(random.randint(1, 31))
            month = str(random.randint(1, 12))
            year = str(random.randint(1900, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 5)
            string_date = month + self.separate_lst[separate1] + date + self.separate_lst[separate2] + year
            result = ""
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(date)): result += "d"
            result += "s"
            for j in range(len(year)): result += "y"

        for i in range(50):
            date = random.randint(1, 31)
            if date < 10:
                date = "0" + str(date)
            else:
                date = str(date)
            month = str(random.randint(1, 12))
            year = str(random.randint(1900, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 5)
            string_date = month + self.separate_lst[separate1] + date + self.separate_lst[separate2] + year
            result = ""
            result += "mm"
            result += "s"
            result += "dd"
            result += "s"
            result += "yyyy"

        return

    # yyyy,mm,dd
    def type_3(self) -> list:
        lst_rtn = []
        for i in range(50):
            date = str(random.randint(1, 31))
            month = str(random.randint(1, 12))
            year = str(random.randint(1900, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 5)
            string_date = year + self.separate_lst[separate1] + month + self.separate_lst[separate2] + date
            result = ""
            for j in range(len(year)): result += "y"
            result = "s"
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(date)): result += "d"

        for i in range(50):
            date = random.randint(1, 31)
            if date < 10:
                date = "0" + str(date)
            else:
                date = str(date)
            month = str(random.randint(1, 12))
            year = str(random.randint(1900, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 5)
            string_date = year + self.separate_lst[separate1] + month + self.separate_lst[separate2] + date
            result = ""
            result += "yyyy"
            result += "s"
            result += "mm"
            result += "s"
            result += "dd"
        return

    # mmm, dd, yyyy
    def type_4(self) -> list:
        date = random.randint(1, 31)
        month = random.randint(1, 12)
        year = random.randint(1900, 2100)
        return

    # yyyy, mmm, dd
    def type_4(self) -> list:
        date = random.randint(1, 31)
        month = random.randint(0, 11)
        year = random.randint(1900, 2100)
        month_eng = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dece"]
        return

    # dd, mmmmm, yyyy
    def type_3(self) -> list:
        date = random.randint(1, 31)
        month = random.randint(0, 11)
        year = random.randint(1900, 2100)
        month_eng = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        return

