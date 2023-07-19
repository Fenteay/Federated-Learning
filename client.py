import os
import threading
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import socket
import pickle
import mysql.connector
import csv
warnings.filterwarnings('ignore')

sync = threading.Event()

def getdata():
    host = "172.31.9.10"
    port = 3306
    user = "root"
    password = "rasp1234"
    database = "diabetes"
    # Kết nối đến MySQL
    connection = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database )
    cursor = connection.cursor()
    query = "SELECT * FROM diabetes"
    cursor.execute(query)
    result = cursor.fetchall()
    # Lấy danh sách các tên cột từ đối tượng cursor
    columns = [i[0] for i in cursor.description]
    # Tạo DataFrame từ kết quả truy vấn và tên cột
    df = pd.DataFrame(result, columns=columns)
    csv_file = 'data.csv'
    # Mở tệp CSV và ghi dữ liệu vào tệp
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i[0] for i in cursor.description])  
        writer.writerows(result) 
    df =pd.read_csv("data.csv")
    cursor.close()
    connection.close()
    return df
def connect_server(server_ip, server_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((server_ip, server_port))
    return server_socket

def send_weights(server_socket, filename):
    print("Sending weights...")
    buffer_size = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        serialized_model = f.read()

    # Gửi buffer size trước
    server_socket.send(str(buffer_size).encode())
    # Chờ xác nhận từ client
    server_socket.recv(1024) 

    # Gửi file pkl
    server_socket.sendall(serialized_model)
    sync.set()
    print("Weights sent successfully.")

def receive_weights(server_socket, filename):
    print("Receiving weights...")
    # Nhận buffer size
    buffer_size = int(server_socket.recv(1024).decode())
    # Gửi xác nhận cho server
    server_socket.send(b"OK")

    # Nhận file pkl
    received_data = b""
    while len(received_data) < buffer_size:
        data = server_socket.recv(buffer_size - len(received_data))
        if not data:
            break
        received_data += data

    with open(filename, 'wb') as f:
        f.write(received_data)
    sync.set()
    print("Weights received successfully.")
    return filename

def perform_training(X_train, y_train, loaded_model):
    trained_weights = loaded_model.fit(X_train, y_train)
    weights_path = "C_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(trained_weights, f)
    return weights_path
if __name__ == "__main__":
    # Load dữ liệu
    df = pd.read_csv("datatest.csv")
    # Xử lý dữ liệu
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    # Phân chia dữ liệu train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Kết nối với server
    server_socket = connect_server('192.168.120.23', 8205)
    # Nhận weight gốc từ server
    file_path = "weight_I.pkl"
    initial_weights = receive_weights(server_socket,file_path)
    # Đợi cho cả luồng nhận hoàn thành
    sync.wait()
    sync.clear()
    with open(initial_weights, "rb") as f:
        loaded_model = pickle.load(f)
    # Train mô hình và gửi lại weight đã train cho server
    trained_weights = perform_training(X_train, y_train, loaded_model)
    send_weights(server_socket,trained_weights)
    # Đợi cho cả luồng gửi hoàn thành
    sync.wait()
    sync.clear()
    # Đóng kết nối socket
    server_socket.close()
