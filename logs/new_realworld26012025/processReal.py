
# files=["avg_decision_delay_neptune(real_world_varyWorkload_Topology50servers).txt",
#        "avg_decision_delay_vsvbp(real_world_varyWorkload_Topology50servers).txt",
#        "avg_decision_delay_cr_eua(real_world_varyWorkload_Topology50servers).txt",
#        "avg_decision_delay_mcf(real_world_varyWorkload_Topology50servers).txt",
#        "avg_decision_delay_pluto(real_world_varyWorkload_Topology50servers).txt",
#        "avg_decision_delay_heu_xu(real_world_varyWorkload_Topology50servers).txt"]
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

    # # print("data =", data)
    #
    # # data = [
    # #     [1.000000, 0.630000],
    # #     [2.000000, 1.782500],
    # #     [3.000000, 5.350000],
    # #     [4.000000, 11.196250],
    # #     [5.000000, 22.511000],
    # #     [6.000000, 60.791667],
    # #     [7.000000, 94.645714],
    # #     [8.000000, 141.622500],
    # #     [9.000000, 208.672778],
    # #     [10.000000, 219.005000]
    # # ]
    #
    # # Extract the second column and calculate the average
    # second_column = [value[1] for value in data]
    # # sum_apps=0
    # # for i in range(len(data)):
    # #  sum_apps= sum_apps+data[i][0]
    # average = sum(second_column) / len(second_column)
    # print(average)

# requests = 100
# files=["delays_neptune(varyApps1-10-50servers).txt",
#        "delays_vsvbp(varyApps1-10-50servers).txt",
#        "delays_cr_eua(varyApps1-10-50servers).txt",
#        "delays_mcf(varyApps1-10-50servers).txt",
#        "delays_pluto(varyApps1-10-50servers).txt",
#        "delays_heu_xu(varyApps1-10-50servers).txt"]
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
#     first_colum = [value[0] for value in data]
#     # Extract the second column and calculate the average
#     second_column = [value[1] for value in data]
#
#     aux_data=[]
#     for i in range(len(first_colum)):
#         aux_data.append(second_column[i]/first_colum[i])
#     average = sum(aux_data) / (len(second_column)*requests)
#     print(average)

workload= [[927, 601, 856, 610, 535, 736, 496, 1116, 492, 403],
[1011, 626, 896, 664, 582, 753, 553, 1188, 533, 443],
[978, 614, 899, 627, 561, 739, 552, 1195, 515, 420],
[937, 590, 876, 598, 545, 716, 537, 1209, 492, 405],
[987, 600, 923, 661, 551, 744, 565, 1267, 473, 411],
[991, 563, 899, 625, 554, 751, 583, 1281, 487, 431],
[983, 568, 911, 612, 541, 708, 574, 1226, 460, 404],
[951, 559, 892, 573, 503, 688, 572, 1254, 468, 393],
[945, 558, 979, 613, 566, 707, 580, 1274, 497, 432],
[957, 590, 955, 617, 564, 693, 587, 1268, 499, 419],
[967, 553, 943, 615, 582, 698, 604, 1240, 521, 433],
[939, 555, 931, 624, 562, 703, 631, 1242, 502, 450]]

w_aux=[]
for i in range(len(workload)):
    w_aux.append(sum(workload[i]))
print(w_aux)

files=["delays_neptune(real_world_varyWorkload_Topology50servers).txt",
       "delays_vsvbp(real_world_varyWorkload_Topology50servers).txt",
       "delays_cr_eua(real_world_varyWorkload_Topology50servers).txt",
       "delays_mcf(real_world_varyWorkload_Topology50servers).txt",
       "delays_pluto(real_world_varyWorkload_Topology50servers).txt",
       "delays_heu_xu(real_world_varyWorkload_Topology50servers).txt"]

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
        aux_data.append(second_column[i]/w_aux[i])
    average = sum(aux_data) / (len(second_column))
    print(average)