import unittest
import nestpy
import platform

import numpy as np

class ConstructorTest(unittest.TestCase):
    """Test constructors

    These are used in the setup of later tests.  Therefore, seperate test
    here.
    """

    def test_vdetector_constructor(self):
        detector = nestpy.detectors.VDetector()
        assert detector is not None
        assert isinstance(detector, nestpy.detectors.VDetector)

    def test_vdetector_initialization(self):
        detector = nestpy.detectors.VDetector()
        detector.Initialization()
        assert detector is not None
        assert isinstance(detector, nestpy.detectors.VDetector)

    def test_xenon_example_constructor(self):
        detector = nestpy.detectors.DetectorExample_XENON10()
        assert detector is not None
        assert isinstance(detector, nestpy.detectors.DetectorExample_XENON10)

    def test_nestcalc_constructor_vdetect(self):
        detector = nestpy.detectors.VDetector()
        detector.Initialization()
        nestcalc = nestpy.NESTcalc(detector)
        assert nestcalc is not None
        assert isinstance(nestcalc, nestpy.NESTcalc)

    def test_intteraction_type_constructor(self):
        it = nestpy.interactions.NR
        assert it is not None
        assert str(it) != ""
        assert isinstance(it, nestpy.interactions)


class VDetectorTest(unittest.TestCase):

   @classmethod
   def setUpClass(cls):
       cls.detector = nestpy.detectors.VDetector()
       cls.detector.Initialization()
       cls.it = nestpy.interactions.NR
       cls.nestcalc = nestpy.NESTcalc(cls.detector)
       cls.nuisance = cls.nestcalc.default_nr_parameters
       cls.free = cls.nestcalc.default_nr_er_width_parameters
       cls.nestcalc = nestpy.NESTcalc(cls.detector)
   # def test_fit_s1(self):
   #     self.detector.FitS1(1.0, 2.0, 3.0)

   def test_fit_ef(self):
       self.detector.FitEF(1.0, 2.0, 3.0)

   # def test_fit_s2(self):
   #     self.detector.FitS2(1.0, 2.0, 3.0)

   def test_fit_tba(self):
       self.detector.FitTBA(1.0, 2.0, 3.0)


class NESTcalcTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = nestpy.detectors.VDetector()
        cls.detector.Initialization()
        cls.it = nestpy.interactions.NR
        cls.nestcalc = nestpy.NESTcalc(cls.detector)

        cls.nuisance = cls.nestcalc.default_nr_parameters
        cls.free = cls.nestcalc.default_nr_er_width_parameters
        cls.er_params = cls.nestcalc.default_er_parameters

    def test_nestcalc_full_calculation(self):
        result = self.nestcalc.FullCalculation(self.it, 1., 2., 3., 4, 5,
                                               self.nuisance,
                                               self.free,
                                               self.er_params,
                                               False)
        assert isinstance(result, nestpy.NESTresult)

    def test_nestcalc_get_photon_times(self):
        self.nestcalc.GetPhotonTimes(self.it, 10, 10, 10., 10.)

    def test_nestcalc_get_yields(self):
        yields = self.nestcalc.GetYields(
            self.it, 10., 10., 10., 10., 10., self.nuisance)

    def test_nestcalc_get_yields_defaults(self):
        yields = self.nestcalc.GetYields(nestpy.interactions.NR,
                                         10)

    def test_nestcalc_get_yields_named(self):
        yields = self.nestcalc.GetYields(nestpy.interactions.NR,
                                         energy=10)

    # def test_nestcalc_get_spike(self):
    #     # This is stalling some builds. Need to improe the test.
    #     self.nestcalc.GetSpike(10, 10., 20., 30., 10., 10., [0, 1, 2])
    
    def test_nestcalc_get_yield_ER_weighted(self):
        self.nestcalc.GetYieldERWeighted(energy=5.2, 
                                         density=2.9, 
                                         drift_field=124, 
                                        )
    
    def test_nestcalc_calculate_g2(self):
        assert self.nestcalc.CalculateG2(True)[3] > 10

    def test_nestcalc_set_drift_velocity(self):
        self.nestcalc.SetDriftVelocity(190, 10, 10, 1.8)

    def test_nestcalc_set_drift_velocity_non_uniform(self):
        self.nestcalc.SetDriftVelocity_NonUniform(2.9, 1, 170, 1.8, 0., 0.)

    def test_nestcalc_set_denisty(self):
        self.nestcalc.SetDensity(190, 10)

    def test_nestcalc_photon_energy(self):
        self.nestcalc.PhotonEnergy(True, True, 190)

    def test_nestcalc_calc_electron_LET(self):
        # shouldn't have to set third argument..
        # but not a used feature by many so non-urgent to solve
        self.nestcalc.CalcElectronLET(100., 54, True) # energy, atom num.(Xe), CSDA

    def test_nest_calc_get_detector(self):
        self.nestcalc.GetDetector()

    def test_equality(self):
        # Will call a test for the nearlyEqual function to ensure it still works.
        self.nestcalc.GetYields(nestpy.interactions.NR, 100., 2.9, 100., 0., 54, nestpy.default_nr_parameters, nestpy.default_er_parameters)


class TestSpectraWIMPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = nestpy.spectra
    
    def test_WIMP_spectrum(self):
        self.spec.WIMP_prep_spectrum( 50., 10. ) #mass and energy integration step
        self.spec.WIMP_spectrum( self.spec.WIMP_prep_spectrum( 50., 10. ), 50., 0. )


class NESTcalcFullCalculationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = nestpy.detectors.VDetector()
        cls.detector.Initialization()
        cls.it = nestpy.interactions.NR

        cls.nestcalc = nestpy.NESTcalc(cls.detector)
        cls.nuisance = cls.nestcalc.default_nr_parameters
        cls.free = cls.nestcalc.default_nr_er_width_parameters
        cls.er_params = cls.nestcalc.default_er_parameters
        cls.result = cls.nestcalc.FullCalculation(
            cls.it, 10., 3., 100., 131, 56,
            cls.nuisance,
            cls.free,
            cls.er_params,
            True)

        cls.position = [2,3,4]

    def test_nestcalc_add_photon_transport_time(self):
       # print(self.result.photon_times)
       self.nestcalc.AddPhotonTransportTime(
           self.result.photon_times, 1.0, 2.0, 3.0)

    def test_nestcalc_get_quanta(self):
        self.nestcalc.GetQuanta(self.result.yields, 10., self.free)

    def test_nestcalc_get_quanta_defaults(self):
        self.nestcalc.GetQuanta(self.result.yields)

    def test_nestcalc_get_s1(self):
        self.nestcalc.GetS1(self.result.quanta,
                            10., 10., -30.,
                            10., 10., -30.,
                            10., 10.,
                            self.it,
                            100, 10., 10.,
                            nestpy.S1CalculationMode.Full, False,
                            [0, 1, 2],
                            [0., 1., 2.])

    def test_nestcalc_get_s2(self):
        self.nestcalc.GetS2(self.result.quanta.electrons, #int ne
                            10., 10., -30., #truth pos x y z
                            10., 10., -30., #smear pos x y z
                            10., 10.,
                            100, 10.,
                            nestpy.S2CalculationMode.Full, False,
                            [0, 1, 2],
                            [0., 1., 2.],
                            [0., 82., 2., 3., 4.])

    def test_nestcalc_get_xyresolution(self):
        self.detector = nestpy.detectors.DetectorExample_XENON10()
        self.detector.Initialization()
        self.nestcalc = nestpy.NESTcalc(self.detector)
        self.nestcalc.xyResolution(
                            0., 1.,2.)

class LArNESTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = nestpy.detectors.VDetector()
        cls.detector.Initialization()
        cls.it = nestpy.LArInteraction(0)

        cls.larnest = nestpy.LArNEST(cls.detector)
        cls.result = cls.larnest.full_calculation(
            cls.it, 100., 1., 500., 1.393, True
        )
    
    def test_larnest_get_yields(self):
        self.larnest.get_yields(self.it, 100., 1., 500., 1.393)

class RunNESTvecParallelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = nestpy.detectors.DetectorExample_XENON10()
        cls.it = nestpy.interactions.NR
        n = 2500
        cls.energies = [1. + 49. * i / n for i in range(n)]
        cls.positions = [[0., 0., 10. + 100. * i / n] for i in range(n)]
        cls.fields = [k for k in dir(nestpy.array.NESTObservableArray) if not k.startswith("_")]

    def run_parallel(self, energies=None, positions=None, **kwargs):
        result = nestpy.array.runNESTvec_parallel(
            self.detector, self.it,
            self.energies if energies is None else energies,
            self.positions if positions is None else positions,
            seed=5, chunk_size=100, **kwargs)
        return {k: getattr(result, k) for k in self.fields}

    def test_independent_of_thread_count(self):
        reference = self.run_parallel(n_threads=1)
        for n_threads in [2, 4, 0]:
            self.assertEqual(self.run_parallel(n_threads=n_threads), reference)

    def test_order_and_length(self):
        full = self.run_parallel(n_threads=4)
        first_chunk = self.run_parallel(self.energies[:100], self.positions[:100], n_threads=4)
        for k in self.fields:
            self.assertEqual(len(full[k]), len(self.energies))
            self.assertEqual(full[k][:100], first_chunk[k])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.run_parallel(positions=self.positions[:-1])
        with self.assertRaises(ValueError):
            nestpy.array.runNESTvec_parallel(
                self.detector, self.it, self.energies, self.positions, chunk_size=0)

    def test_seed_is_uint64(self):
        for run in [nestpy.array.runNESTvec, nestpy.array.runNESTvec_parallel]:
            run(self.detector, self.it, self.energies[:10], self.positions[:10], seed=2**64 - 1)
            with self.assertRaises(TypeError):
                run(self.detector, self.it, self.energies[:10], self.positions[:10], seed=-1)

    def test_run_nest_threads(self):
        arr = nestpy.helpers.run_nest(
            self.it, self.detector, self.energies, np.array(self.positions),
            n_threads=2, seed=5, chunk_size=100)
        self.assertEqual(arr["s1c_phd"].to_list(), self.run_parallel(n_threads=1)["s1c_phd"])

if __name__ == "__main__":
    unittest.main()
