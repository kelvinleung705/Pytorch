import csv
import random


class GenerateDate:
    separate_lst = ["/", ",", "-", " ", "'"]
    mon_eng = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dece"]
    month_eng = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    def __init__(self):
        self.csv_list = []
    # dd,mm,yyyy
    def type_1(self) -> list:
        lst_rtn = []
        for i in range(50):
            day = str(random.randint(1, 31))
            month = str(random.randint(1, 12))
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = day + self.separate_lst[separate1] + month + self.separate_lst[separate2] + year
            result = ""
            for j in range(len(day)): result += "d"
            result += "s"
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(year)): result += "y"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        for i in range(50):
            day = random.randint(1, 31)
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)
            month = random.randint(1, 12)
            if month < 10:
                month = "0" + str(month)
            else:
                month = str(month)
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = day + self.separate_lst[separate1] + month + self.separate_lst[separate2] + year
            result = ""
            result += "dd"
            result += "s"
            result += "mm"
            result += "s"
            result += "yyyy"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        return lst_rtn

    # mm,dd,yyyy
    def type_2(self) -> list:
        lst_rtn = []
        for i in range(50):
            day = str(random.randint(1, 31))
            month = str(random.randint(1, 12))
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = month + self.separate_lst[separate1] + day + self.separate_lst[separate2] + year
            result = ""
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(day)): result += "d"
            result += "s"
            for j in range(len(year)): result += "y"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        for i in range(50):
            day = random.randint(1, 31)
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)
            month = random.randint(1, 12)
            if month < 10:
                month = "0" + str(month)
            else:
                month = str(month)
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = month + self.separate_lst[separate1] + day + self.separate_lst[separate2] + year
            result = ""
            result += "mm"
            result += "s"
            result += "dd"
            result += "s"
            result += "yyyy"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        return lst_rtn


    # yyyy,mm,dd
    def type_3(self) -> list:
        lst_rtn = []
        for i in range(50):
            day = str(random.randint(1, 31))
            month = str(random.randint(1, 12))
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = year + self.separate_lst[separate1] + month + self.separate_lst[separate2] + day
            result = ""
            for j in range(len(year)): result += "y"
            result += "s"
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(day)): result += "d"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        for i in range(50):
            day = random.randint(1, 31)
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)
            month = random.randint(1, 12)
            if month < 10:
                month = "0" + str(month)
            else:
                month = str(month)
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = year + self.separate_lst[separate1] + month + self.separate_lst[separate2] + day
            result = ""
            result += "yyyy"
            result += "s"
            result += "mm"
            result += "s"
            result += "dd"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        return lst_rtn


    # mmm, dd, yyyy
    def type_4(self) -> list:
        lst_rtn = []
        for i in range(50):
            day = str(random.randint(1, 31))
            month = random.randint(0, 11)
            month = self.mon_eng[month]
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = month + self.separate_lst[separate1] + day + self.separate_lst[separate2] + year
            result = ""
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(month)): result += "d"
            result += "s"
            for j in range(len(day)): result += "y"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        for i in range(50):
            day  = random.randint(1, 31)
            if day  < 10:
                day  = "0" + str(day)
            else:
                day  = str(day )
            month = random.randint(0, 11)
            month = self.mon_eng[month]
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = month + self.separate_lst[separate1] + day + self.separate_lst[separate2] + year
            result = ""
            result += "mmm"
            result += "s"
            result += "dd"
            result += "s"
            result += "yyyy"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        return lst_rtn


    # dd, mmm, yyyy
    def type_5(self) -> list:
        lst_rtn = []
        for i in range(50):
            day = str(random.randint(1, 31))
            month = random.randint(0, 11)
            month = self.mon_eng[month]
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = day + self.separate_lst[separate1] + month + self.separate_lst[separate2] + year
            result = ""
            for j in range(len(month)): result += "d"
            result += "s"
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(day)): result += "y"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        for i in range(50):
            day = random.randint(1, 31)
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)
            month = random.randint(0, 11)
            month = self.mon_eng[month]
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = day + self.separate_lst[separate1] + month + self.separate_lst[separate2] + year
            result = ""
            result += "dd"
            result += "s"
            result += "mmm"
            result += "s"
            result += "yyyy"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        return lst_rtn



    # yyyy, mmm, dd
    def type_6(self) -> list:
        lst_rtn = []
        for i in range(50):
            day = str(random.randint(1, 31))
            month = random.randint(0, 11)
            month = self.mon_eng[month]
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = year + self.separate_lst[separate1] + month + self.separate_lst[separate2] + day
            result = ""
            for j in range(len(month)): result += "y"
            result += "s"
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(day)): result += "d"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        for i in range(50):
            day = random.randint(1, 31)
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)
            month = random.randint(0, 11)
            month = self.mon_eng[month]
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = year + self.separate_lst[separate1] + month + self.separate_lst[separate2] + day
            result = ""
            result += "yyyy"
            result += "s"
            result += "mmm"
            result += "s"
            result += "dd"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        return lst_rtn


    # dd, mmmmm, yyyy
    def type_7(self) -> list:
        lst_rtn = []
        for i in range(50):
            day = str(random.randint(1, 31))
            month = random.randint(0, 11)
            month = self.month_eng[month]
            year = str(random.randint(2000, 2100))
            separate1 = random.randint(0, 4)
            separate2 = random.randint(0, 4)
            string_date = day + " " + month + "," + year
            result = ""
            for j in range(len(day)): result += "d"
            result += "s"
            for j in range(len(month)): result += "m"
            result += "s"
            for j in range(len(year)): result += "y"
            lst = [string_date]
            lst.extend(list(result))
            lst_rtn.append(lst)

        return lst_rtn

    def produce_csv(self, name="date_set"):
        self.csv_list.extend(self.type_1())
        self.csv_list.extend(self.type_2())
        self.csv_list.extend(self.type_3())
        self.csv_list.extend(self.type_4())
        self.csv_list.extend(self.type_5())
        self.csv_list.extend(self.type_6())
        self.csv_list.extend(self.type_7())
        with open(name+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.csv_list)

if __name__ == '__main__':
    datesys = GenerateDate()
    datesys.produce_csv()






