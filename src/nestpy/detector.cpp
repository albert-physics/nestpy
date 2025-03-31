#include <pybind11/pybind11.h>

#include "detector.hh"

#include <iostream>
#include "NEST.hh"
#include "VDetector.hh"
#include "LUX_Run03.hh"
#include "LZ_SR1.hh"
#include "DetectorExample_XENON10.hh"

namespace py = pybind11;

void init_detector(py::module& m){

    auto m_detect = m.def_submodule("detectors", "detectors");

    //	Binding for example XENON10
    // py::class_<DetectorExample_XENON10, VDetector, std::unique_ptr<DetectorExample_XENON10, py::nodelete>>(m_detect, "DetectorExample_XENON10")
    // 	.def(py::init<>())
    // 	.def("Initialization", &DetectorExample_XENON10::Initialization)
    // 	.def("FitTBA", &DetectorExample_XENON10::FitTBA)
    // 	.def("OptTrans", &DetectorExample_XENON10::OptTrans)
    // 	.def("SinglePEWaveForm", &DetectorExample_XENON10::SinglePEWaveForm);
    
    // Binding for example LUX_Run03
    // py::class_<DetectorExample_LUX_RUN03, VDetector, std::unique_ptr<DetectorExample_LUX_RUN03, py::nodelete>>(m_detect, "LUX_Run03")
    // 	.def(py::init<>())
    // 	.def("Initialization", &DetectorExample_LUX_RUN03::Initialization)
    // 	.def("FitTBA", &DetectorExample_LUX_RUN03::FitTBA)
    // 	.def("OptTrans", &DetectorExample_LUX_RUN03::OptTrans)
    // 	.def("SinglePEWaveForm", &DetectorExample_LUX_RUN03::SinglePEWaveForm);

    //  Binding for LUX-ZEPLIN's WS2022 result
    py::class_<LZ_Detector, VDetector, std::unique_ptr<LZ_Detector, py::nodelete>>(m_detect, "LZ_WS2022")
        .def(py::init<>())
        .def("Initialization", &LZ_Detector::Initialization)
        .def("FitTBA", &LZ_Detector::FitTBA)
        .def("OptTrans", &LZ_Detector::OptTrans)
        .def("SinglePEWaveForm", &LZ_Detector::SinglePEWaveForm);
}