import os
from os.path import join
import numpy as np
from laspy.file import File
from ismember import ismember


class GeneralDataset():
	def __init__(self, files, num_dims, cube_size=10, npoints=4096, split='pred', seed=np.random.randint(0,100), sigma=0.0, zyx_max_angle=[0,0,0], overlap=0):
		'''
		Inicialization of the Dataset.

		:param files: files.
		:param num_dims: Number of dimensions of the dataset.
		:param cube_size: size of the cube that is fed to the network in meters.
		:param npoints: Number of points to feed the network.
		:param split: split that this dataset belogns to, (train, test, pred). [Default = 'pred'].
		:param seed: seed for random processes. [Default = np.random.randint(0,100)].
		:param sigma: standar deviation of the normal distribution used to augment train data.
		:param zyx_max_angle: ± maximum angles applied for data augmnetation in Z, Y and X. [Default = [0,0,0]].
		:param overlap: min overlap between cubes in pred and test. [Default = 0].
		'''

		self.files = files
		self.num_dims = num_dims
		self.cube_size = cube_size
		self.npoints = npoints
		self.sigma = sigma
		self.zyx_max_angle = zyx_max_angle
		self.split = split
		self.overlap = overlap
		
		#List with files that belong to the dataset
		
		# Seed
		self.np_RandomState = np.random.RandomState(seed)


	def __in_cube(self, coordinates, centre, n_points):
		"""
		Function that returns an array with a len equal to self.npoints with the index corresponding to points in
		a cube of self.cube_size long centred in centre. The cube extends in Z dimension.

		:param coordinates: xyz.
		:param centre: xyz coordinates of the centre of the cube.
		:param n_points: number of points selected. If np.nan it returns all [Default = np.nan].
		:return 
			n_idx_in_cube: array with npoints indexes corresponding to points in the cube (replace if there are not enough).
		"""

		# Get boundaries of the cube. 
		min_coords = centre-[self.cube_size/2,self.cube_size/2,self.cube_size/2]
		max_coords = centre+[self.cube_size/2,self.cube_size/2,self.cube_size/2]

		# To make sure that it is not leaving data on the Z axis.
		# Modify those values to the limit of the cloud
		min_coords[2] = np.min(coordinates,axis=0)[2]
		max_coords[2] = np.max(coordinates,axis=0)[2]

		# Take points that are inside of the cube.
		idx_in_cube = np.sum((coordinates>=min_coords)*(coordinates<=max_coords),axis=1)==3
		idx_in_cube = np.where(idx_in_cube)[0]

		# From those points, pick self.npoints randomly.
		if np.isnan(n_points): return idx_in_cube

		if len(coordinates) >= n_points:
			choice = self.np_RandomState.choice(len(idx_in_cube), n_points, replace=False)
		else:
			choice = self.np_RandomState.choice(len(idx_in_cube), n_points, replace=True)

		idx_in_cube = idx_in_cube[choice]

		return idx_in_cube


	def __cubes_through_the_cloud(self, coordinates):
		'''
		Function to calculated the indexes of the cubes through the coordinates. The point cloud is divied in cubes.
		The cubes has a minimum overlap equal to self.overlap between them. The point cloud is divided first in X and then in Y dimension.
		The first cube and the last are located in the extrems. The inner cubes are positioned so that they all have the same overlapping.
		Points are randomly selected.

		:param coordinates: coordinates of the point cloud.
		:return indexes: array n x n_points with the indexes of the selected points in the n cubes.
		'''
	
		# Calculate the number of cubes and their centre
		# n * c = L + S_u (n-1) -> n = ceil((L - S_u_min) / (c - S_u_min)) ; S_u = (n * C - L) / (n - 1)
		centres = np.array([]).reshape(-1,3)
		
		# Calculate the centre coordinates of each cube.

		# split in X
		max_loc = coordinates[:,0].max()
		min_loc = coordinates[:,0].min()
		l = max_loc - min_loc # length in X
		n = np.ceil((l - self.overlap) / (self.cube_size - self.overlap)) # number of cubes
		overlap = (n * self.cube_size - l) / (n - 1) if n > 1 else 0 # recalculate overlap
		centres_x = min_loc + self.cube_size/2 + np.arange(n) * (self.cube_size - overlap) # X locations of the centres

		# split in Y
		for centre_x in centres_x:
			# Points between this X positions
			this_coordinates = np.logical_and(coordinates[:,0] >= (centre_x - self.cube_size/2), coordinates[:,0] <= (centre_x + self.cube_size/2))
			
			# Calculate The number of cubes and their centres.
			max_loc = coordinates[this_coordinates,1].max()
			min_loc = coordinates[this_coordinates,1].min()
			l = max_loc - min_loc
			n = np.ceil((l - self.overlap) / (self.cube_size - self.overlap)).astype('int')
			overlap = (n * self.cube_size - l) / (n - 1) if n > 1 else 0

			centres_y = np.zeros((n,3))
			centres_y[:,0] = centre_x # X centre is the same for all. Z does not care because the cube is expanded in Z.
			centres_y[:,1] = min_loc + self.cube_size/2 + np.arange(n) * (self.cube_size - overlap) # Y centres

			# Append these centres with the others
			centres = np.append(centres, centres_y, axis=0)

		# Calcualte the indexes of the n points of each cube.
		
		#indexes of the points of each cube.
		indexes = np.zeros((len(centres), self.npoints), dtype='int')
		no_empty= np.zeros((len(centres)), dtype='bool')

		for i in range(len(centres)):

			# indexes in this cube
			idx_in_cube = self.__in_cube(coordinates, centres[i], n_points=np.nan)
			# If there are no point continue
			if ~np.any(idx_in_cube):
				continue
			else:
				no_empty[i] = True

			# If there are more points than necesary
			if len(idx_in_cube) >= self.npoints:
				choice = self.np_RandomState.choice(len(idx_in_cube), self.npoints, replace=False)
			else:
				choice = self.np_RandomState.choice(len(idx_in_cube), self.npoints, replace=True)

			indexes[i] = idx_in_cube[choice]

		# Remove empty points
		indexes = indexes[no_empty]

		return indexes


	def __getitem__(self, index):
		'''
		Function to load a point cloud from the dataset. It load a cube with N points for train and val split,
		or M cubes with N cubes with the overlap indicated covering all the point cloud.
		Train and val split have data aumentation.

		:param index: Index of file that will be loaded.
		return:
			raw_point_set: raw point clouds with the selected points. Nx3. (len(indexes)x3 if split = test or pred).
			final_point_set: normalised point cloud with the selected points. Nx3 (MxNx3 if split = test or pred)
			final_semantic_seg: semantic labels. Nx3. (len(indexes)x3 if split = test or pred).
			final_instance_seg: instance labels. Nx3. (len(indexes)x3 if split = test or pred).
			indexes: if splt = test or pred, indexes of the final_point_set in raw_point_set.
		'''

		# Get file names to load
		file_name=self.files[index]

		# Use laspy to load the file
		inFile = File(file_name, mode = "r")

		#Load coordinates and move to avoid having values too high
		coordinates = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
		# coordinates = coordinates - ((np.amax(coordinates,0)+np.amin(coordinates,0))/2)

		if self.split == 'train':
			"""
			Select random points inside a cube centre in a random points.
			The data are augmented by applying Z-rotations.
			The data is normalised. Labels are returned.
			"""

			# Load labels of each point
			semantic_seg = (inFile.Classification).astype(np.int32)
			instance_seg = (inFile.user_data).astype(np.int32)
			semantic_seg_nodes = (inFile.nodes).astype(np.int32)
			semantic_seg[semantic_seg_nodes==1] = semantic_seg.max()+1 # labels of nodes

			# Choose a random point as center of the cube that will be taken
			curcenter = coordinates[self.np_RandomState.choice(len(coordinates),1)[0],:]
			# Calculate n_points inside the cube
			curchoice = self.__in_cube(coordinates, curcenter, n_points=self.npoints)

			# Seletc points
			raw_point_set = coordinates[curchoice,:]
			final_semantic_seg = semantic_seg[curchoice]
			final_instance_seg = instance_seg[curchoice]

			# Add random normal noise
			raw_point_set += self.np_RandomState.normal(0, self.sigma, (len(raw_point_set),3))

			# Move points to the origin and resize to values [0,1]
			final_point_set = (raw_point_set - np.amin(raw_point_set, axis = 0))/self.cube_size

			# Rotate point cloud in zyx random angles
			zyx = [self.np_RandomState.uniform()*self.zyx_max_angle[0]*2 - self.zyx_max_angle[0],
				self.np_RandomState.uniform()*self.zyx_max_angle[1]*2 - self.zyx_max_angle[1],
				self.np_RandomState.uniform()*self.zyx_max_angle[2]*2 - self.zyx_max_angle[2]]

			cv_z = np.cos(zyx[0])
			sv_z = np.sin(zyx[0])
			cv_y = np.cos(zyx[1])
			sv_y = np.sin(zyx[1])
			cv_x = np.cos(zyx[2])
			sv_x = np.sin(zyx[2])

			rotation_matrix = np.array([[cv_y*cv_z, -cv_x*sv_z + sv_y*cv_z*sv_x,  sv_x*sv_z + sv_y*cv_z*cv_x],
										[cv_y*sv_z,  cv_x*cv_z + sv_y*sv_z*sv_x, -sv_x*cv_z + sv_y*sv_z*cv_x],
										[-sv_y,      cv_y*sv_x,                   cv_x*cv_y]])

			final_point_set = np.dot(rotation_matrix, final_point_set.transpose()).transpose()

			return raw_point_set, final_point_set, final_semantic_seg, final_instance_seg
				

		elif self.split == 'test':
			"""
			Divide the point cloud in cubes covering the whole cloud.
			Select random points inside each cube.
			"""

			# Load labels of the point
			semantic_seg = (inFile.Classification).astype(np.int32)
			instance_seg = (inFile.user_data).astype(np.int32)
			semantic_seg_nodes = (inFile.nodes).astype(np.int32)
			semantic_seg[semantic_seg_nodes==1] = semantic_seg.max()+1 # labels of nodes
			
			# Split the point cloud in cubes
			indexes = self.__cubes_through_the_cloud(coordinates)
			

			raw_point_set = coordinates[indexes]
			final_semantic_seg = semantic_seg[indexes]
			final_instance_seg = instance_seg[indexes]

			return raw_point_set, final_semantic_seg, final_instance_seg					

		elif self.split == 'pred':
			"""
			Divide the point cloud in cubes covering the whole cloud.
			Select random points inside each cube.
			"""
			
			# Split the point cloud in cubes
			indexes = self.__cubes_through_the_cloud(coordinates)
		
			raw_point_set = coordinates[indexes]

			return raw_point_set

	def get_batch(self,start_idx, end_idx):
		'''
		Function used to get batches for a training step. It returns the data between start_idx and end_idx.

		:param start_idx:first file.
		:param end_idx: last file
		:return:
			batch_data_raw: raw point clouds with the selected points.
			batch_data: normalised point clouds wiht the selected points.
			batch_label_seg: semantic values.
			batch_label_inst: instance values.
		'''
		
		if self.split =='train':
			# The batch is made up of point clouds of different files.
			bsize = end_idx - start_idx
			batch_data_raw = np.zeros((bsize, self.npoints, self.num_dims), dtype=np.float32)
			batch_data = np.zeros((bsize, self.npoints, self.num_dims), dtype=np.float32)
			batch_label_seg = np.zeros((bsize, self.npoints), dtype=np.int32)
			batch_label_inst = np.zeros((bsize, self.npoints), dtype=np.int32)

			for i in range(bsize):
				raw_ps, ps, seg, inst = self[start_idx+i]
				batch_data_raw[i,...] = raw_ps
				batch_data[i,...] = ps
				batch_label_seg[i,:] = seg
				batch_label_inst[i,:] = inst

			return batch_data_raw, batch_data, batch_label_seg, batch_label_inst


	def __len__(self):
		return len(self.files)