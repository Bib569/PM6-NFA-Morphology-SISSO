program centroid_distances
    implicit none
    ! Constants
    integer, parameter :: n_polymer = 500
    integer, parameter :: n_nfa = 600
    integer, parameter :: atoms_per_polymer = 314
    integer, parameter :: atoms_per_nfa = 217
    real, parameter :: cutoff = 6.0  ! Angstrom
    integer, parameter :: n_bins = 60 ! For histogram
    real, parameter :: bin_width = 0.1 ! Angstrom
    ! Variables
    real, allocatable :: coords(:,:,:)  ! (x/y/z, atom_id, molecule_id)
    real, allocatable :: polymer_centroids(:,:)  ! (x/y/z, polymer_id)
    real, allocatable :: nfa_centroids(:,:)      ! (x/y/z, nfa_id)
    real, allocatable :: distances(:)            ! Array to store distances
    integer, allocatable :: histogram(:)         ! Array for histogram
    real :: box(3)                              ! Box dimensions
    integer :: i, j, k, n_distances, ierr
    real :: dx, dy, dz, dist
    ! Allocate arrays
    allocate(polymer_centroids(3,n_polymer), stat=ierr)
    if (ierr /= 0) then
        write(*,*) 'Error allocating polymer_centroids'
        stop
    endif
    allocate(nfa_centroids(3,n_nfa), stat=ierr)
    if (ierr /= 0) then
        write(*,*) 'Error allocating nfa_centroids'
        stop
    endif
    allocate(distances(n_polymer*n_nfa), stat=ierr)
    if (ierr /= 0) then
        write(*,*) 'Error allocating distances'
        stop
    endif
    allocate(histogram(n_bins), stat=ierr)
    if (ierr /= 0) then
        write(*,*) 'Error allocating histogram'
        stop
    endif
    ! Read coordinates from trajectory file
    call read_trajectory()
    write(*,*) 'Calculating polymer centroids...'
    ! Calculate centroids for polymers
    do i = 1, n_polymer
        do k = 1, 3
            polymer_centroids(k,i) = sum(coords(k,:,i)) / atoms_per_polymer
        end do
    end do
    write(*,*) 'Calculating NFA centroids...'
    ! Calculate centroids for NFAs
    do i = 1, n_nfa
        do k = 1, 3
            nfa_centroids(k,i) = sum(coords(k,:,i+n_polymer)) / atoms_per_nfa
        end do
    end do
    ! Calculate distances
    write(*,*) 'Calculating distances...'
    n_distances = 0
    histogram = 0
    do i = 1, n_polymer
        do j = 1, n_nfa
            ! Calculate minimum image distance (assuming periodic boundary conditions)
            dx = polymer_centroids(1,i) - nfa_centroids(1,j)
            dy = polymer_centroids(2,i) - nfa_centroids(2,j)
            dz = polymer_centroids(3,i) - nfa_centroids(3,j)
            ! Apply minimum image convention
            dx = dx - box(1) * nint(dx/box(1))
            dy = dy - box(2) * nint(dy/box(2))
            dz = dz - box(3) * nint(dz/box(3))
            dist = sqrt(dx*dx + dy*dy + dz*dz)
            ! Store distances within cutoff
            if (dist <= cutoff) then
                n_distances = n_distances + 1
                distances(n_distances) = dist
                ! Add to histogram
                k = int(dist/bin_width) + 1
                if (k <= n_bins) histogram(k) = histogram(k) + 1
            end if
        end do
    end do
    ! Output results
    call write_results()
    ! Deallocate arrays
    deallocate(coords, polymer_centroids, nfa_centroids, distances, histogram)
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
    subroutine write_results()
        implicit none
        integer :: i
        real :: bin_center
        ! Write distances
        open(unit=10, file='centroid_distances.dat', status='replace', iostat=ierr)
        if (ierr /= 0) then
            write(*,*) 'Error opening centroid_distances.dat'
            stop
        endif
        write(*,*) 'Writing distances to file...'
        write(*,*) 'Total distances within cutoff:', n_distances
        do i = 1, n_distances
            write(10,*) distances(i)
        end do
        close(10)
        ! Write histogram
        open(unit=11, file='distance_histogram.dat', status='replace', iostat=ierr)
        if (ierr /= 0) then
            write(*,*) 'Error opening distance_histogram.dat'
            stop
        endif
        write(*,*) 'Writing histogram to file...'
        do i = 1, n_bins
            bin_center = (i-0.5) * bin_width
            write(11,*) bin_center, histogram(i)
        end do
        close(11)
    end subroutine write_results
end program centroid_distances
