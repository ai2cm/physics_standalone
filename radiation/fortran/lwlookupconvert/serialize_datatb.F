program test

    use physparam, only : kind_phys
    use module_radlw_avplank, only : totplnk
      
    implicit none

      print *, 'Serializing out data...'
      write(*,*) 'kind_phys = ', kind_phys


!$ser init directory='./saveData' prefix='radlw_avplnk'
!$ser mode write      
!$ser verbatim print *, "Serializing requested data"
!$ser savepoint 'mySavepoint'
!$ser data totplnk=totplnk
!$ser cleanup

end program test
      
