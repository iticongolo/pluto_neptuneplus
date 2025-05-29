
# files=["avg_network_delay_neptune(varyApps1-10-50servers).txt",
#        "avg_network_delay_vsvbp(varyApps1-10-50servers).txt",
#        "avg_network_delay_cr_eua(varyApps1-10-50servers).txt",
#        "avg_network_delay_mcf(varyApps1-10-50servers).txt",
#        "avg_network_delay_pluto(varyApps1-10-50servers).txt",
#        "avg_network_delay_heu_xu(varyApps1-10-50servers).txt"]
#
# for i in range(len(files)):
#     data = []
#     with open(files[i], "r") as file:
#         for line in file:
#             # Skip empty lines
#             if not line.strip():
#                 continue
#             # Split the line into parts
#             parts = line.split()
#             # Convert the parts to floats
#             x = float(parts[0])
#             y = float(parts[1])
#             # Apply the transformation on y
#             data.append([x, y])
#
#     # print("data =", data)
#
#     # data = [
#     #     [1.000000, 0.630000],
#     #     [2.000000, 1.782500],
#     #     [3.000000, 5.350000],
#     #     [4.000000, 11.196250],
#     #     [5.000000, 22.511000],
#     #     [6.000000, 60.791667],
#     #     [7.000000, 94.645714],
#     #     [8.000000, 141.622500],
#     #     [9.000000, 208.672778],
#     #     [10.000000, 219.005000]
#     # ]
#
#     # Extract the second column and calculate the average
#     second_column = [value[1] for value in data]
#     # sum_apps=0
#     # for i in range(len(data)):
#     #  sum_apps= sum_apps+data[i][0]
#     average = sum(second_column) / len(second_column)
#     print(average)

requests = 100
files=["delays_neptune(varyApps1-10-50servers).txt",
       "delays_vsvbp(varyApps1-10-50servers).txt",
       "delays_cr_eua(varyApps1-10-50servers).txt",
       "delays_mcf(varyApps1-10-50servers).txt",
       "delays_pluto(varyApps1-10-50servers).txt",
       "delays_heu_xu(varyApps1-10-50servers).txt"]
for i in range(len(files)):
    data = []
    with open(files[i], "r") as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            # Split the line into parts
            parts = line.split()
            # Convert the parts to floats
            x = float(parts[0])
            y = float(parts[1])
            # Apply the transformation on y
            data.append([x, y])

    first_colum = [value[0] for value in data]
    # Extract the second column and calculate the average
    second_column = [value[1] for value in data]

    aux_data=[]
    for i in range(len(first_colum)):
        aux_data.append(second_column[i]/first_colum[i])
    average = sum(aux_data) / (len(second_column)*requests)
    print(average)
