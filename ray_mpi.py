from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def ray_tracing(x, y):
    pixel = np.array([x, y, 0])
    origin = camera
    direction = normalize(pixel - origin)
    color = np.zeros((3))
    reflection = 1
    for k in range(max_depth):
        nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
        if nearest_object is None:
            break
        intersection = origin + min_distance * direction
        normal_to_surface = normalize(intersection - nearest_object['center'])
        shifted_point = intersection + 1e-5 * normal_to_surface
        intersection_to_light = normalize(light['position'] - shifted_point)
        _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
        intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
        is_shadowed = min_distance < intersection_to_light_distance
        if is_shadowed:
            break
        illumination = np.zeros((3))
        illumination += nearest_object['ambient'] * light['ambient']
        illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)
        intersection_to_camera = normalize(camera - intersection)
        H = normalize(intersection_to_light + intersection_to_camera)
        illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
        color += reflection * illumination
        reflection *= nearest_object['reflection']
        origin = shifted_point
        direction = reflected(direction, normal_to_surface)
    return color

start_time = MPI.Wtime()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

max_depth = 3
width = 3840
height = 2160
camera = np.array([0, 0, 1])
light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
objects = [
    # Sun
    { 'center': np.array([-4.5, -4, -3]), 'radius': 3, 'ambient': np.array([1, 0.5, 0.31]), 'diffuse': np.array([1, 0.5, 0.31]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Sun
    #Mercury
    { 'center': np.array([-2, -2, -3]), 'radius': 0.14, 'ambient': np.array([0.8, 0.8, 0.8]), 'diffuse': np.array([0.8, 0.8, 0.8]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Mercury
    #Venus
    { 'center': np.array([-1.45, -1.5, -3]), 'radius': 0.21, 'ambient': np.array([0.9, 0.6, 0.3]), 'diffuse': np.array([0.9, 0.6, 0.3]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Venus
    #Earth
    { 'center': np.array([-0.8, -1, -3]), 'radius': 0.245, 'ambient': np.array([0.1, 0.5, 0.8]), 'diffuse': np.array([0.1, 0.5, 0.8]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Earth
    #Mars
    { 'center': np.array([-0.3, -0.5, -3]), 'radius': 0.175, 'ambient': np.array([0.9, 0.3, 0.3]), 'diffuse': np.array([0.9, 0.3, 0.3]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Mars
    #Jupiter
    { 'center': np.array([0.55, 0.31, -3]), 'radius': 0.49, 'ambient': np.array([0.8, 0.7, 0.5]), 'diffuse': np.array([0.8, 0.7, 0.5]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Jupiter
    #Saturn
    { 'center': np.array([1.7, 1, -3]), 'radius': 0.42, 'ambient': np.array([0.9, 0.8, 0.5]), 'diffuse': np.array([0.9, 0.8, 0.5]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Saturn
    #Uranus
    { 'center': np.array([2.7, 1.4, -3]), 'radius': 0.28, 'ambient': np.array([0.6, 0.8, 0.9]), 'diffuse': np.array([0.6, 0.8, 0.9]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 },  # Uranus
    #Neptune
    { 'center': np.array([3.5, 2, -3]), 'radius': 0.245, 'ambient': np.array([0.3, 0.5, 0.8]), 'diffuse': np.array([0.3, 0.5, 0.8]), 'specular': np.array([0.5, 0.5, 0.5]), 'shininess': 32, 'reflection': 0.5 }   # Neptune
]

# 작업 분배
chunk_sizes = [(height // size) + (1 if i < height % size else 0) for i in range(size)]
start_rows = [sum(chunk_sizes[:i]) for i in range(size)]
end_rows = [start_rows[i] + chunk_sizes[i] for i in range(size)]

local_start_row = start_rows[rank]
local_end_row = end_rows[rank]
local_height = local_end_row - local_start_row

# 화면 비율 설정
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)

# 각 프로세스의 지역 이미지 저장
local_image = np.empty((local_height, width, 3))

# 화면 좌표 배열 생성
Y = np.linspace(screen[1], screen[3], height)
X = np.linspace(screen[0], screen[2], width)

# 할당된 행에 대해 레이 트레이싱 수행
for i in range(local_height):
    y = Y[local_start_row + i]
    for j in range(width):
        x = X[j]
        local_image[i, j] = ray_tracing(x, y)

# 결과 수집 준비
gathered_image = None
if rank == 0:
    gathered_image = np.empty((height, width, 3))

# Gatherv를 위한 recvcounts와 displacements 계산
recvcounts = np.array(chunk_sizes) * width * 3
displacements = np.array(start_rows) * width * 3

# 각 프로세스의 결과를 루트 프로세스로 모으기
comm.Gatherv(np.clip(local_image, 0, 1), [gathered_image, recvcounts, displacements, MPI.DOUBLE], root=0)

# 최종 이미지를 루트 프로세스에서 저장
if rank == 0:
    plt.imsave('image.png', gathered_image)

# 전체 수행 시간 출력
end_time = MPI.Wtime()
print(f"Overall elapsed time: {end_time - start_time}, Rank: {rank}")
