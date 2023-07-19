import os
import threading
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import socket
import pickle

sync = threading.Event()
# Kết nối với client
def connect_client(client_ip, client_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((client_ip, client_port))
    return client_socket


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
    
def aggregate_weights(weight1, weight2):
    # Trích xuất thông tin cấu trúc và tham số từ weight1
    n_neighbors_1 = weight1.n_neighbors
    metric_1 = weight1.metric
    p_1 = weight1.p

    # Trích xuất thông tin cấu trúc và tham số từ weight2
    n_neighbors_2 = weight2.n_neighbors
    metric_2 = weight2.metric
    p_2 = weight2.p

    # Tính toán trọng số tổng hợp
    aggregated_n_neighbors = (n_neighbors_1 + n_neighbors_2) // 2
    aggregated_metric = metric_1 if metric_1 == metric_2 else None
    aggregated_p = p_1 if p_1 == p_2 else None

    # Tạo đối tượng KNeighborsClassifier với trọng số tổng hợp
    aggregated_weights = KNeighborsClassifier(
        n_neighbors=aggregated_n_neighbors,
        metric=aggregated_metric,
        p=aggregated_p
    )

    return aggregated_weights
# Tạo kết nối socket và gửi/nhận weights từ client
def federated_learning(best_model):
    hostname = socket.gethostname()
    server_ip = socket.gethostbyname(hostname)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_ip, 8000))
    server_socket.listen(2) 
    print("Waiting for clients to connect... from " + server_ip)
    client1_socket, client1_addr = server_socket.accept()
    print("Connected to client 1:", client1_addr)
    client2_socket, client2_addr = server_socket.accept()
    print("Connected to client 2:", client2_addr)
    send_weights(client1_socket, best_model)
    send_weights(client2_socket, best_model)
    sync.wait()
    sync.clear()       
    receive_weights(client1_socket, "C1_weights.pkl")
    receive_weights(client2_socket, "C2_weights.pkl")
    sync.wait()
    sync.clear()
    with open("C1_weights.pkl", "rb") as f:
        weight = pickle.load(f)
    with open("C2_weights.pkl", "rb") as f:
        weight2 = pickle.load(f)
    updated_weights = aggregate_weights(weight,weight2)
    weights_s = "S_weights.pkl"
    with open(weights_s, "wb") as f:
        pickle.dump(updated_weights, f)
    # Đóng kết nối
    client1_socket.close()
    client2_socket.close()
    server_socket.close()

if __name__ == "__main__":
    df = pd.read_csv("diabetes.csv")
    df=df.drop_duplicates()
    df.isnull().sum()
    target_name='Outcome'
    y= df[target_name]
    X=df.drop(target_name,axis=1)
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)
    knn= KNeighborsClassifier()
    n_neighbors = list(range(15,25))
    p=[1,2]
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    hyperparameters = dict(n_neighbors=n_neighbors, p=p,weights=weights,metric=metric)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1',error_score=0)
    best_model = grid_search.fit(X_train,y_train)
    knn_pred = best_model.predict(X_test)
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)  # Huấn luyện lại trên toàn bộ dữ liệu
    weights_path = "knn_model_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(best_model, f)
    # Chạy mô hình Federated Learning
    federated_learning(weights_path)
