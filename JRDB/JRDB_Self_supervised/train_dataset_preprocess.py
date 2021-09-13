import json
import numpy as np
import torch
import open3d as o3d
import random
from random import randint
import open3d as o3d
print('no_random_drop')
print('random_translation')

def create_frame_detections(frame):
	objects=frame['objects']
	frame_id=frame['frame_id']
	keys=['id','name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score', 'frame_idx', 'fix_count']
	detection={}
	for key in keys:
		detection[key]=[]
	obj_len=len(objects)
	ignore_obj=[]
	#if obj_len>80:
	#    print('object_increased 80' )
	#random.shuffle(objects)
	'''
	Drop=random.choice([True,False])
	if Drop:
		if obj_len>20:
			n_obj=random.choice([3,4])
			if n_obj==3:
				ignore_obj=[randint(1,5),randint(6,11),randint(12,obj_len)]
				obj_len=obj_len-3
			elif n_obj==4:
				ignore_obj=[randint(1,4),randint(5,9),randint(10,15),randint(16,obj_len)]
				obj_len=obj_len-4
		elif obj_len>10:
			n_obj=random.choice([1,2])
			if n_obj==1:
				ignore_obj=[randint(1,obj_len)]
				obj_len=obj_len-1
			elif n_obj==2:
				ignore_obj=[randint(1,6),randint(7,obj_len)]
				obj_len=obj_len-2
				'''
	detection['name']=np.array([[1] for i in range(obj_len)])
	detection['occluded']=np.array([-1 for i in range(obj_len)],dtype='float32')
	detection['truncated']=np.array([-1 for i in range(obj_len)],dtype='float32')
	detection['alpha']=np.array([-1 for i in range(obj_len)],dtype='float32')
	detection['bbox']=np.array([[-1,-1,-1,-1] for i in range(obj_len)],dtype='float32')
	detection['score']=np.array([-1 for i in range(obj_len)],dtype='float32')
	detection['fix_count']=np.array([-1 for i in range(obj_len)],dtype='float32')
	detection['frame_idx']=frame_id.split('.')[0]

	bbox_3D_keys=['cy', 'l', 'rot_z', 'h', 'w', 'cx', 'cz']
	i=0
	for object_ in objects:
		bbox_3D=object_['box']
		object_id=object_['label_id'].split(':')[1]
		detection['id'].append([object_id])
		object_['label_id'].split(':')[1]
		#[x,y,z]=[-bbox_3D['cy'], -bbox_3D['cz'] + bbox_3D['h']/2, bbox_3D['cx']]
		#rotation_y = (-bbox_3D['rot_z'] if bbox_3D['rot_z'] < np.pi else 2 * np.pi - bbox_3D['rot_z'])
		detection['dimensions'].append([bbox_3D['l'],bbox_3D['h'],bbox_3D['w']])
		detection['location'].append([bbox_3D['cx'],bbox_3D['cy'],bbox_3D['cz']])
		detection['rotation_y'].append(bbox_3D['rot_z'])

	detection['dimensions']=np.array(detection['dimensions'],dtype='float32')
	detection['location']=np.array(detection['location'],dtype='float32')
	detection['rotation_y']=np.array(detection['rotation_y'],dtype='float32')
	detection['id']=np.array(detection['id'],dtype='float32')

	return detection

def get_pointcloud(dets,seq_id,frame_id,root_dir):
	pt_cloud=torch.load(root_dir+seq_id+'/depth/'+frame_id+'.pt')
	extracted_points=[]
	bbox_points = []
	points_split = [0]
	loc = dets["location"].copy() # This is in the camera coordinates
	dims = dets["dimensions"].copy() # This should be standard lhw(camera) format
	rots = dets["rotation_y"].copy()
	boxes = np.concatenate(
		[loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
	#print(boxes)
	for box in boxes:
		center=box[0:3]
		e=box[3:6]
		#e=np.array([e[0],e[1],e[2]])
		e=np.array([e[1],e[0],e[2]])

		rotation=box[6]

		rotation_matrix = np.array([
			[-np.sin(rotation), np.cos(rotation), 0.0],
			[0.0, 0.0, 1.0],
			[np.cos(rotation), np.sin(rotation), 0.0]
			])
		a=o3d.geometry.OrientedBoundingBox()
		center=center.astype(np.float64)
		rotation=rotation.astype('float64')
		e=e.astype('float64')

		#oriented_bbox=o3d.cpu.pybind.geometry.OrientedBoundingBox(center, rotation_matrix, e)
		oriented_bbox=o3d.pybind.geometry.OrientedBoundingBox(center, rotation_matrix, e)
		bbox_alighned=oriented_bbox.get_axis_aligned_bounding_box()
		max_1=oriented_bbox.get_max_bound()
		min_1=oriented_bbox.get_min_bound()
		[maxX,maxY,maxZ]=max_1
		[minX,minY,minZ]=min_1
		lidar=pt_cloud
		mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
						(lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
						(lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))

		random_cutout=random.choice(['left','right','None'])
		if random_cutout=='left':
			minX_2=float(minX+((maxX-minX)/4))
			minY_2=float(minY+((maxY-minY)/4))
			mask = np.where((lidar[:, 0] >= minX_2) & (lidar[:, 0] <= maxX) &
					(lidar[:, 1] >= minY_2) & (lidar[:, 1] <= maxY) &
					(lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))

		elif random_cutout=='right':
			maxX_2=float(maxX-((maxX-minX)/4))
			maxY_2=float(maxY-((maxY-minY)/3))
			mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX_2) &
					(lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY_2) &
					(lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))

		else:
			mask = np.where((lidar[:, 0] >= minX) & (lidar[:, 0] <= maxX) &
				(lidar[:, 1] >= minY) & (lidar[:, 1] <= maxY) &
				(lidar[:, 2] >= minZ) & (lidar[:, 2] <= maxZ))
			
		bbox_point=lidar[mask]
		


		#print('bbox_point',bbox_point)
		if bbox_point.shape[0] == 0:
			bbox_point = np.zeros(shape=(1,4))

		random_translate_x = random.choice([0.1,0.2,0.4])
		random_translate_y = random.choice([0.1,0.2,0.4])
		random_translate_z = random.choice([0.1,0.2,0.4])
		#random_scale=random.choice([0.95,1.05])
		#a=np.ones((12,3))
		flip=False
		#bbox_point=bbox_point*random_scale
		bbox_point=bbox_point+[random_translate_x,random_translate_y,random_translate_z,0]



		#print('bbox_point_type',bbox_point[:,0:3].shape)
		pcd_for_rotation = o3d.geometry.PointCloud()
		pcd_for_rotation.points = o3d.utility.Vector3dVector(bbox_point[:,0:3])
		center=pcd_for_rotation.get_center()
		rot_angle=random.choice([0, np.pi/36, np.pi/18, -np.pi/36, -np.pi/18])
		rotate_object_z = np.array([
			[np.cos(rot_angle), np.sin(rot_angle), 0.0],
			[-np.sin(rot_angle), np.cos(rot_angle), 0.0],
			[0.0, 0.0, 1.0]
			])
		pcd_for_rotation.rotate(rotate_object_z, center)
		bbox_point[:,0:3]=np.asarray(pcd_for_rotation.points)
		
		if flip:
			bbox_point=bbox_point*[1,-1,1,1]

		points_split.append(points_split[-1]+bbox_point.shape[0])
		bbox_points.append(bbox_point)

	bbox_points = np.concatenate(bbox_points, axis=0)
	example = {
		'points': bbox_points[:,0:3],
		'points_split': points_split,
	}
	return example

def get_frame_det_info():
	frame_det_info = {}
	frame_det_info.update({
		'rot': [],
		'loc': [],
		'dim': [],
		'points': [],
		'points_split': [],
		'info_id': [],
	})
	return frame_det_info

def get_val_names():
	val=[
		'clark-center-2019-02-28_1',
		'gates-ai-lab-2019-02-08_0',
		'huang-2-2019-01-25_0',
		'meyer-green-2019-03-16_0',
		'nvidia-aud-2019-04-18_0',
		'tressider-2019-03-16_1',
		'tressider-2019-04-26_2'
		]
	return val


def get_train_names():

	train = [
		'bytes-cafe-2019-02-07_0',
		'clark-center-2019-02-28_0',
		'clark-center-2019-02-28_1',
		'clark-center-intersection-2019-02-28_0',
		'cubberly-auditorium-2019-04-22_0',
		'forbes-cafe-2019-01-22_0',
		'gates-159-group-meeting-2019-04-03_0',
		'gates-ai-lab-2019-02-08_0',
		'gates-basement-elevators-2019-01-17_1',
		'gates-to-clark-2019-02-28_1',
		'hewlett-packard-intersection-2019-01-24_0',
		'huang-2-2019-01-25_0',
		'huang-basement-2019-01-25_0',
		'huang-lane-2019-02-12_0',
		'jordan-hall-2019-04-22_0',
		'memorial-court-2019-03-16_0',
		'meyer-green-2019-03-16_0',
		'nvidia-aud-2019-04-18_0',
		'packard-poster-session-2019-03-20_0',
		'packard-poster-session-2019-03-20_1',
		'packard-poster-session-2019-03-20_2',
		'stlc-111-2019-04-19_0',
		'svl-meeting-gates-2-2019-04-08_0',
		'svl-meeting-gates-2-2019-04-08_1',
		'tressider-2019-03-16_0',
		'tressider-2019-03-16_1',
		'tressider-2019-04-26_2'
	]
	val=get_val_names()
	train=[train[i] for i in range(len(train)) if train[i] not in val]
	return train