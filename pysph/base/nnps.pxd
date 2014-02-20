# numpy
cimport numpy as np

# PyZoltan CArrays
from pyzoltan.core.carray cimport UIntArray, IntArray, DoubleArray, LongArray

# local imports
from particle_array cimport ParticleArray
from point cimport *

cdef inline int real_to_int(double val, double step)
cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size)

cpdef UIntArray arange_uint(int start, int stop=*)

# Basic particle array wrapper used for NNPS
cdef class NNPSParticleArrayWrapper:
    cdef public DoubleArray x,y,z,h
    cdef public UIntArray gid
    cdef public IntArray tag
    cdef public ParticleArray pa
    cdef str name
    cdef int np
    
    # get the number of particles
    cdef int get_number_of_particles(self)

# Domain limits for the simulation
cdef class DomainLimits:
    cdef public double xmin, xmax
    cdef public double ymin, ymax
    cdef public double zmin, zmax

    cdef public double xtranslate
    cdef public double ytranslate
    cdef public double ztranslate

    cdef public int dim
    cdef public bint periodic_in_x, periodic_in_y, periodic_in_z
    cdef public bint is_periodic

# Cell to hold particle indices
cdef class Cell:
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef cIntPoint _cid                 # Spatial index for the cell
    cdef public bint is_boundary        # Flag to indicate boundary cells
    cdef int narrays                    # Number of particle arrays
    cdef public list lindices           # Local indices for particles
    cdef public list gindices           # Global indices for binned particles
    cdef list nparticles                # Number of particles in the cell
    cdef double cell_size               # bin size
    cdef public cPoint centroid         # Centroid computed from indices
    cdef cPoint boxmin                  # Bounding box min for the cell
    cdef cPoint boxmax                  # Bounding box max for the cell
    cdef int layers                     # Layers to compute bounding box
    cdef IntArray nbrprocs              # List of neighboring processors
    cdef public int size                # total number of particles in this cell

    ############################################################################
    # Member functions
    ############################################################################
    # set the indices for the cell
    cpdef set_indices(self, int index, UIntArray lindices, UIntArray gindices)

    # compute the bounding box for a cell. Layers is used to determine
    # the factor times the cell size the bounding box is offset from
    # the cell.
    cdef _compute_bounding_box(self, double cell_size,
                               int layers)

# Nearest neighbor locator
cdef class NNPS:
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public bint trace               # Flag for timing and debugging
    cdef public list particles           # list of particle arrays
    cdef public list pa_wrappers         # list of particle array wrappers
    cdef public int narrays              # Number of particle arrays

    cdef bint in_parallel                # Flag to determine if in parallel
    cdef public object comm              # MPI communicator object
    cdef public int rank                 # MPI rank
    cdef public int size                 # MPI size

    cdef public DomainLimits domain      # Domain limits for the geometry
    cdef public bint is_periodic         # flag for periodicity

    cdef int dim                         # Dimensionality of the problem
    cdef public double cell_size         # Cell size for binning
    cdef public double radius_scale      # Radius scale for kernel

    ############################################################################
    # Member functions
    ############################################################################
    # Main binning routine for NNPS for local particles. This clears
    # the current cell data, re-computes the cell size and bins all
    # particles locally.
    cpdef update(self)

    # Index particles given by a list of indices. The indices are
    # assumed to be of type unsigned int and local to the NNPS object
    cdef _bin(self, int pa_index, UIntArray indices)

    # Compute the cell size across processors. The cell size is taken
    # as max(h)*radius_scale
    cdef _compute_cell_size(self)

    # Neighbor query function. Returns the list of neighbors for a
    # requested particle. The returned list is assumed to be of type
    # unsigned int to follow the type of the local and global ids.
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs)

    # Testing function for brute force neighbor search. The return
    # list is of the same type of the local and global ids (uint)
    cpdef brute_force_neighbors(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs)

# NNPS using the original gridding algorithm
cdef class BoxSortNNPS(NNPS):
    cdef public dict cells               # lookup table for the cells

# NNPS using the linked list approach
cdef class LinkedListNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef DoubleArray xmin                # co-ordinate min values
    cdef DoubleArray xmax                # co-ordinate max values
    cdef IntArray ncells                 # number of cells in each direction
    cdef IntArray cell_shifts            # cell shifts
    cdef public int ncells_tot                  # total number of cells
    cdef public bint fixed_h             # Constant cell sizes
    cdef public list heads               # Head arrays for the cells
    cdef public list nexts               # Next arrays for the particles

    ############################################################################
    # Member functions
    ############################################################################
    # refresh head and next arrays
    cpdef _refresh(self)

    # get neighbors for a cell
    cpdef get_cell_neighbors(
        self, int cell_index, int pa_index, UIntArray nbrs)

    # filter the potential neighbors for a particle given the cell
    # neighbors
    cpdef get_nearest_particles_by_cell(
        self, int src_index, int dst_index, int d_idx, UIntArray potential_nbrs,
        UIntArray nbrs)

    # get the indices for a given cell
    cpdef get_cell_indices(
        self, int cell_index, int pa_index, UIntArray indices)
