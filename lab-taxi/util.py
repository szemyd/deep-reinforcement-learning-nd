def writeToCsv(myCsvRow):
    with open('diagnostics.csv', 'a') as fd:
        fd.write(myCsvRow)