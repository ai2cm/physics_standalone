program test

    use physparam, only : kind_phys
    use module_radsw_cldprtb, only : extliq1, extliq2, ssaliq1, ssaliq2, &
      asyliq1, asyliq2, extice2, ssaice2, asyice2, extice3, ssaice3, &
      asyice3, fdlice3, abari, bbari, cbari, dbari, ebari, fbari, &
      b0r, c0r, b0s, b1s, c0s
      
    implicit none

      print *, 'Serializing out data...'
      write(*,*) 'kind_phys = ', kind_phys


!$ser init directory='./saveData' prefix='radsw_cldprtb'
!$ser mode write      
!$ser verbatim print *, "This is a test"
!$ser savepoint 'mySavepoint'
!$ser data extliq1=extliq1 extliq2=extliq2 ssaliq1=ssaliq1 ssaliq2=ssaliq2
!$ser data asyliq1=asyliq1 asyliq2=asyliq2 extice2=extice2 ssaice2=ssaice2
!$ser data asyice2=asyice2 extice3=extice3 ssaice3=ssaice3 asyice3=asyice3
!$ser data fdlice3=fdlice3 abari=abari bbari=bbari cbari=cbari dbari=dbari
!$ser data ebari=ebari fbari=fbari b0r=b0r c0r=c0r b0s=b0s b1s=b1s c0s=c0s      
!$ser cleanup

end program test
      
