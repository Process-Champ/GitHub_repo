def test_sheet():
    sheet = connect_sheet()
    sheet.append_row(["TEST", "TEST", "TEST", "BUY", 100, 1, "OPEN", 0, "test"])
    print("✅ Test row added")

test_sheet()
