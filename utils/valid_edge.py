import numpy as np
import torch

def is_intersect(edge1, edge2):
    lon1, lat1, lon2, lat2 = edge1
    lon3, lat3, lon4, lat4 = edge2
    distance1_3 = abs(lon1 - lon3) * 100000 + abs(lat1 - lat3) * 100000
    distance1_4 = abs(lon1 - lon4) * 100000 + abs(lat1 - lat4) * 100000
    distance2_3 = abs(lon2 - lon3) * 100000 + abs(lat2 - lat3) * 100000
    distance2_4 = abs(lon2 - lon4) * 100000 + abs(lat2 - lat4) * 100000
    min_distance = np.min([distance1_3, distance1_4, distance2_3, distance2_4])
    if min_distance == 0:
        return False
    else:
        if np.max([lon1, lon2]) < np.min([lon3, lon4]) or np.max([lon3, lon4]) < np.min([lon1, lon2]):
            return False
        else:
            sort_points = np.sort([lon1, lon2, lon3, lon4])
            left_point, right_point = sort_points[1], sort_points[2]
            if lon1 == lon2:
                value_point1 = [lat1, lat2]
            else:
                value_point1 = [(lat2-lat1)/(lon2-lon1)*(left_point-lon1)+lat1, (lat2-lat1)/(lon2-lon1)*(right_point-lon1)+lat1]
            if lon3 == lon4:
                value_point2 = [lat3, lat4]
            else:
                value_point2 = [(lat4 - lat3) / (lon4 - lon3) * (left_point - lon3) + lat3,
                               (lat4 - lat3) / (lon4 - lon3) * (right_point - lon3) + lat3]
            if np.max(value_point1) < np.min(value_point2) or np.max(value_point2) < np.min(value_point1):
                return False
            else:
                return True


def is_acute(edge1, edge2):
    lon1, lat1, lon2, lat2 = edge1
    lon3, lat3, lon4, lat4 = edge2
    distance1_3 = abs(lon1-lon3)*100000 + abs(lat1-lat3)*100000
    distance1_4 = abs(lon1-lon4)*100000 + abs(lat1-lat4)*100000
    distance2_3 = abs(lon2-lon3)*100000 + abs(lat2-lat3)*100000
    distance2_4 = abs(lon2-lon4)*100000 + abs(lat2-lat4)*100000
    min_distance = np.min([distance1_3, distance1_4, distance2_3, distance2_4])
    if min_distance > 0:
        return False
    else:
        if distance1_3 == min_distance:
            x1,y1 = lon2-lon1, lat2-lat1
            x2,y2 = lon4-lon3, lat4-lat3
        if distance1_4 == min_distance:
            x1,y1 = lon2-lon1, lat2-lat1
            x2,y2 = lon3-lon4, lat3-lat4
        if distance2_3 == min_distance:
            x1,y1 = lon1-lon2, lat1-lat2
            x2,y2 = lon4-lon3, lat4-lat3
        if distance2_4 == min_distance:
            x1,y1 = lon1-lon2, lat1-lat2
            x2,y2 = lon3-lon4, lat3-lat4

        vector_1 = [x1, y1]
        vector_2 = [x2, y2]
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) / math.pi * 180
        if angle < 40:
            return True
        else:
            return False


def is_valid(idx, node_coords, edge_index):
    new_edge = torch.flatten(node_coords[idx])
    existing_edges_u = torch.stack([node_coords[idx] for idx in edge_index[0]])
    existing_edges_v = torch.stack([node_coords[idx] for idx in edge_index[1]])
    existing_edges = torch.cat((existing_edges_u, existing_edges_v), dim=-1)
    for existing_edge in existing_edges:
        if is_intersect(new_edge, existing_edge) or is_acute(new_edge, existing_edge):
            return False
    return True