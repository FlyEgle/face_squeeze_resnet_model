import pymysql


def db_connect():
    db = pymysql.connect("localhost", "root", "root", "test")
    cursor = db.cursor()

    cursor.execute("SELECT VERSION()")
    data = cursor.fetchone()
    print("Database Version: %s" % data)

    db.close()

if __name__ == '__main__':

    db_connect()
