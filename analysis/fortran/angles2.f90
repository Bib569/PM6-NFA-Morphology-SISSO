program angle_histogram
    implicit none
    ! Constants
    integer, parameter :: n_polymer = 500
    integer, parameter :: n_nfa = 600
    integer, parameter :: atoms_per_polymer = 314
    integer, parameter :: atoms_per_nfa = 217
    real, parameter :: cutoff = 6.0  ! Angstrom
    integer, parameter :: n_bins = 18 ! For histogram
    real, parameter :: bin_width = 10.0  ! 3 degrees per bin for 60 bins to cover 0-180 degrees

    ! Variables
    real, allocatable :: coords(:,:,:)         ! (x/y/z, atom_id, molecule_id)
    real, allocatable :: polymer_centroids(:,:)  ! (x/y/z, polymer_id)
    real, allocatable :: nfa_centroids(:,:)      ! (x/y/z, nfa_id)
    real, allocatable :: normal_vectors_polymer(:,:) ! (x/y/z, polymer_id)
    real, allocatable :: normal_vectors_nfa(:,:)     ! (x/y/z, nfa_id)
    real, allocatable :: o_angles(:)           ! Array for intersection angles (o)
    real, allocatable :: a_angles(:)           ! Array for intersection angles (a)
    real, allocatable :: histogram_o(:), histogram_a(:)
    integer :: i, j, n_o_angles, n_a_angles, ierr, k
    real :: angle, dx, dy, dz
    real :: box(3)
    real :: backbone_vector_polymer(3, n_polymer)
    real :: backbone_vector_nfa(3, n_nfa)

    ! Allocate arrays (moved to read_trajectory subroutine)
    allocate(polymer_centroids(3, n_polymer))  ! 3 coordinates (x,y,z) for each polymer
    allocate(nfa_centroids(3, n_nfa))         ! 3 coordinates (x,y,z) for each NFA
    
    allocate(normal_vectors_polymer(3, n_polymer), stat=ierr)
    if (ierr /= 0) stop 'Allocation failed for normal_vectors_polymer'
    
    allocate(normal_vectors_nfa(3, n_nfa), stat=ierr)
    if (ierr /= 0) stop 'Allocation failed for normal_vectors_nfa'
    
    allocate(o_angles(n_polymer * n_nfa), stat=ierr)
    if (ierr /= 0) stop 'Allocation failed for o_angles'
    
    allocate(a_angles(n_polymer * n_nfa), stat=ierr)
    if (ierr /= 0) stop 'Allocation failed for a_angles'
    
    allocate(histogram_o(n_bins), stat=ierr)
    if (ierr /= 0) stop 'Allocation failed for histogram_o'
    
    allocate(histogram_a(n_bins), stat=ierr)
    if (ierr /= 0) stop 'Allocation failed for histogram_a'

    ! Initialize arrays
    polymer_centroids = 0.0
    nfa_centroids = 0.0
    normal_vectors_polymer = 0.0
    normal_vectors_nfa = 0.0
    o_angles = 0.0
    a_angles = 0.0
    histogram_o = 0
    histogram_a = 0

    ! Read trajectory file
    call read_trajectory()

    ! Calculate centroids and normal vectors
    call calculate_centroids_and_normals()

    ! Calculate intersection angles
    n_o_angles = 0
    n_a_angles = 0
    do i = 1, n_polymer
        do j = 1, n_nfa
            dx = polymer_centroids(1,i) - nfa_centroids(1,j)
            dy = polymer_centroids(2,i) - nfa_centroids(2,j)
            dz = polymer_centroids(3,i) - nfa_centroids(3,j)

            ! Apply minimum image convention
            dx = dx - box(1) * nint(dx/box(1))
            dy = dy - box(2) * nint(dy/box(2))
            dz = dz - box(3) * nint(dz/box(3))

            if (sqrt(dx**2 + dy**2 + dz**2) <= cutoff) then
                ! Calculate o_angle (angle between normal vectors)
                angle = acos(dot_product(normal_vectors_polymer(:,i), normal_vectors_nfa(:,j)))
                n_o_angles = n_o_angles + 1
                o_angles(n_o_angles) = angle * 180.0 / acos(-1.0)  ! Convert to degrees

                ! Add to o histogram
                k = int(o_angles(n_o_angles)/bin_width) + 1
                if (k >= 1 .and. k <= n_bins) then
                    histogram_o(k) = histogram_o(k) + 1
                endif

                ! Calculate a_angle (angle between backbone vectors)
                angle = acos(abs(dot_product(backbone_vector_polymer(:,i), &
                                           backbone_vector_nfa(:,j))))
                n_a_angles = n_a_angles + 1
                a_angles(n_a_angles) = angle * 180.0 / acos(-1.0)  ! Convert to degrees

                ! Add to a histogram
                k = int(a_angles(n_a_angles)/bin_width) + 1
                if (k >= 1 .and. k <= n_bins) then
                    histogram_a(k) = histogram_a(k) + 1
                endif

                ! Debug output
                if (mod(n_o_angles, 100) == 0) then
                    write(*,*) 'O-angle calculation:', n_o_angles, o_angles(n_o_angles)
                    write(*,*) 'Normal vectors:'
                    write(*,*) 'Polymer:', normal_vectors_polymer(:,i)
                    write(*,*) 'NFA:', normal_vectors_nfa(:,j)
                endif
            end if
        end do
    end do

    ! Write results
    call write_results("o_angles.dat", o_angles, n_o_angles, histogram_o)
    call write_results("a_angles.dat", a_angles, n_a_angles, histogram_a)

    ! After creating histogram, add:
    write(*,*) 'Histogram bin counts:'
    do i = 1, n_bins
        write(*,*) i, histogram_a(i)
    end do

    ! After calculating angles, add:
    write(*,*) 'A-angle range:', minval(a_angles(1:n_a_angles)), maxval(a_angles(1:n_a_angles))
    write(*,*) 'Number of a_angles:', n_a_angles

    ! Add debug prints for both angles
    write(*,*) 'O-angle range:', minval(o_angles(1:n_o_angles)), maxval(o_angles(1:n_o_angles))
    write(*,*) 'Number of o_angles:', n_o_angles
    write(*,*) 'O-angle histogram counts:'
    do i = 1, n_bins
        write(*,*) i, histogram_o(i)
    end do

    ! Deallocate arrays
    deallocate(coords, polymer_centroids, nfa_centroids, normal_vectors_polymer, &
               normal_vectors_nfa, o_angles, a_angles, histogram_o, histogram_a)

    write(*,*) 'Program completed successfully!'

contains

      subroutine read_trajectory()
        implicit none
        integer :: i, j, natoms
        character(len=80) :: title
        character(len=5) :: resid, atomname
        character(len=8) :: coordx, coordy, coordz
        integer :: atomnum
        real :: x, y, z
        logical :: file_exists
        ! Check if trajectory file exists
        inquire(file='prd.gro', exist=file_exists)
        if (.not. file_exists) then
            write(*,*) 'Error: trajectory.gro file not found!'
            stop
        endif
        ! Allocate coordinates array
        if (.not. allocated(coords)) then
            allocate(coords(3, max(atoms_per_polymer, atoms_per_nfa), n_polymer + n_nfa), &
                    stat=ierr)
            if (ierr /= 0) then
                write(*,*) 'Error: Failed to allocate coords array!'
                stop
            endif
        endif
        ! Open trajectory file
        open(unit=12, file='prd.gro', status='old', action='read', iostat=ierr)
        if (ierr /= 0) then
            write(*,*) 'Error opening trajectory file'
            stop
        endif
        ! Read title
        read(12,'(A80)', iostat=ierr) title
        if (ierr /= 0) then
            write(*,*) 'Error reading title'
            stop
        endif
        ! Read number of atoms
        read(12,*, iostat=ierr) natoms
        if (ierr /= 0) then
            write(*,*) 'Error reading number of atoms'
            stop
        endif
        ! Verify number of atoms
        if (natoms /= (n_polymer * atoms_per_polymer + n_nfa * atoms_per_nfa)) then
            write(*,*) 'Error: Number of atoms in file does not match expected total'
            write(*,*) 'Expected:', n_polymer * atoms_per_polymer + n_nfa * atoms_per_nfa
            write(*,*) 'Found:', natoms
            stop
        endif
        write(*,*) 'Reading coordinates...'
        ! Read polymer coordinates
        do i = 1, n_polymer
            do j = 1, atoms_per_polymer
                read(12,'(i5,2a5,i5,3f8.3)', iostat=ierr) atomnum, resid, atomname, &
                    atomnum, x, y, z
                if (ierr /= 0) then
                    write(*,*) 'Error reading coordinates at polymer', i, 'atom', j
                    stop
                endif
                ! Convert from nm to Angstrom
                coords(1,j,i) = x * 10.0
                coords(2,j,i) = y * 10.0
                coords(3,j,i) = z * 10.0
            end do
        end do
        ! Read NFA coordinates
        do i = 1, n_nfa
            do j = 1, atoms_per_nfa
                read(12,'(i5,2a5,i5,3f8.3)', iostat=ierr) atomnum, resid, atomname, &
                    atomnum, x, y, z
                if (ierr /= 0) then
                    write(*,*) 'Error reading coordinates at NFA', i, 'atom', j
                    stop
                endif
                ! Convert from nm to Angstrom
                coords(1,j,i+n_polymer) = x * 10.0
                coords(2,j,i+n_polymer) = y * 10.0
                coords(3,j,i+n_polymer) = z * 10.0
            end do
        end do
        ! Read box dimensions (in nm)
        read(12,*, iostat=ierr) box(1), box(2), box(3)
        if (ierr /= 0) then
            write(*,*) 'Error reading box dimensions'
            stop
        endif
        ! Convert box dimensions from nm to Angstrom
        box = box * 10.0
        close(12)
        write(*,*) 'Successfully read trajectory file'
        write(*,*) 'Box dimensions (Å):', box
    end subroutine read_trajectory

    subroutine calculate_centroids_and_normals()
        implicit none
        integer :: i, j
        real :: v1(3), v2(3)
        ! Calculate centroids and normal vectors for polymers
        do i = 1, n_polymer
            do k = 1, 3
                polymer_centroids(k,i) = sum(coords(k,:,i)) / atoms_per_polymer
            end do
            ! Compute normal vector using first three atoms
            v1 = coords(:,2,i) - coords(:,1,i)
            v2 = coords(:,3,i) - coords(:,1,i)
            normal_vectors_polymer(:,i) = cross_product(v1, v2) / norm(cross_product(v1, v2))
            ! Use atoms that are along the backbone, for example atoms 1 and 10
            backbone_vector_polymer(:,i) = coords(:,10,i) - coords(:,1,i)
            backbone_vector_polymer(:,i) = backbone_vector_polymer(:,i) / norm(backbone_vector_polymer(:,i))
        end do
        ! Calculate centroids and normal vectors for NFAs
        do i = 1, n_nfa
            do k = 1, 3
                nfa_centroids(k,i) = sum(coords(k,:,i+n_polymer)) / atoms_per_nfa
            end do
            ! Compute normal vector using first three atoms
            v1 = coords(:,2,i+n_polymer) - coords(:,1,i+n_polymer)
            v2 = coords(:,3,i+n_polymer) - coords(:,1,i+n_polymer)
            normal_vectors_nfa(:,i) = cross_product(v1, v2) / norm(cross_product(v1, v2))
            ! Use atoms that are along the backbone, for example atoms 1 and 10
            backbone_vector_nfa(:,i) = coords(:,10,i+n_polymer) - coords(:,1,i+n_polymer)
            backbone_vector_nfa(:,i) = backbone_vector_nfa(:,i) / norm(backbone_vector_nfa(:,i))
        end do
        ! In calculate_centroids_and_normals(), add debug prints:
        write(*,*) 'Sample backbone vectors:'
        write(*,*) 'First polymer:', backbone_vector_polymer(:,1)
        write(*,*) 'First NFA:', backbone_vector_nfa(:,1)
    end subroutine calculate_centroids_and_normals

    subroutine write_results(filename, angles, n_angles, histogram)
        implicit none
        character(len=*), intent(in) :: filename
        real, intent(in) :: angles(:)
        integer, intent(in) :: n_angles
        real, intent(in) :: histogram(:)
        integer :: i, max_bin
        real :: bin_center, average_angle, most_probable_angle
        real :: normalized_histogram(n_bins)
        
        ! Calculate average angle
        average_angle = sum(angles(1:n_angles)) / n_angles
        
        ! Find most probable angle (from histogram)
        max_bin = maxloc(histogram, dim=1)
        most_probable_angle = (max_bin - 0.5) * bin_width
        
        ! Normalize histogram
        if (n_angles > 0) then
            normalized_histogram = histogram / (n_angles * bin_width)
        else
            normalized_histogram = 0.0
        endif

        ! Write angles to file
        open(unit=10, file=filename, status='replace', iostat=ierr)
        if (ierr /= 0) then
            write(*,*) 'Error opening ', filename
            stop
        endif
        
        ! Write statistics at the top of the file
        write(10,*) '# Statistics:'
        write(10,*) '# Average angle:', average_angle
        write(10,*) '# Most probable angle:', most_probable_angle
        write(10,*) '# Raw angle data:'
        
        do i = 1, n_angles
            write(10,*) angles(i)
        end do
        close(10)
        
        ! Write normalized histogram with statistics
        open(unit=11, file=trim(filename)//"_histogram.dat", status='replace', iostat=ierr)
        write(11,*) '# Average angle:', average_angle
        write(11,*) '# Most probable angle:', most_probable_angle
        do i = 1, n_bins
            bin_center = (i-0.5) * bin_width
            write(11,*) bin_center, normalized_histogram(i)
        end do
        close(11)
    end subroutine write_results

    function cross_product(a, b) result(c)
        implicit none
        real, intent(in) :: a(3), b(3)
        real :: c(3)
        
        c(1) = a(2)*b(3) - a(3)*b(2)
        c(2) = a(3)*b(1) - a(1)*b(3)
        c(3) = a(1)*b(2) - a(2)*b(1)
    end function cross_product

    function norm(vector) result(magnitude)
        implicit none
        real, intent(in) :: vector(:)
        real :: magnitude
        
        magnitude = sqrt(sum(vector**2))
    end function norm

end program angle_histogram

