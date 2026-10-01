#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

#include "NEST.hh"
#include "execNEST.hh"

namespace py = pybind11;
using namespace pybind11::literals;


// Seed for one chunk. Hashing the user seed first stops nearby seeds sharing
// chunk seeds (with seed + chunk, seed 42 chunk 1 would equal seed 43 chunk 0)
static uint64_t chunk_seed(uint64_t seed, uint64_t chunk){
    return splitmix64(seed) + chunk;
}

// Multithreaded runNESTvec. Events are split into fixed-size chunks, each run
// by runNESTvec with its own seed, so the output depends on the seed and
// chunk_size but not on the number of threads.
NESTObservableArray runNESTvec_parallel(
    VDetector* detector,
    INTERACTION_TYPE particleType,
    const std::vector<double>& eList,
    const std::vector<std::vector<double>>& pos3dxyz,
    double inField,
    uint64_t seed,
    const std::vector<double>& ERYieldsParam,
    const std::vector<double>& NRYieldsParam,
    const std::vector<double>& NRERWidthsParam,
    S1CalculationMode s1mode,
    S2CalculationMode s2mode,
    bool calculate_times,
    int n_threads,
    int chunk_size
){
    if (eList.size() != pos3dxyz.size())
        throw std::invalid_argument("energies and positions must have the same length");
    if (chunk_size <= 0)
        throw std::invalid_argument("chunk_size must be positive");

    const size_t n_events = eList.size();
    const size_t n_chunks = (n_events + chunk_size - 1) / chunk_size;
    if (n_threads <= 0)
        n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = static_cast<int>(std::min<size_t>(n_threads, std::max<size_t>(n_chunks, 1)));

    // Configure the shared detector once, so worker threads only ever read it
    NESTcalc(detector).SetDensity(detector->get_T_Kelvin(), detector->get_p_bar());

    std::vector<NESTObservableArray> chunk_results(n_chunks);
    std::vector<std::exception_ptr> errors(n_threads);
    std::atomic<size_t> next_chunk{0};

    auto worker = [&](int thread_id){
        try {
            for (size_t c = next_chunk++; c < n_chunks; c = next_chunk++){
                const size_t begin = c * chunk_size;
                const size_t end = std::min(begin + chunk_size, n_events);
                chunk_results[c] = runNESTvec(
                    detector, particleType,
                    std::vector<double>(eList.begin() + begin, eList.begin() + end),
                    std::vector<std::vector<double>>(pos3dxyz.begin() + begin, pos3dxyz.begin() + end),
                    inField, chunk_seed(seed, c),
                    ERYieldsParam, NRYieldsParam, NRERWidthsParam,
                    s1mode, s2mode, calculate_times);
            }
        } catch (...) {
            errors[thread_id] = std::current_exception();
            next_chunk = n_chunks;  // stop the other workers early
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t)
        threads.emplace_back(worker, t);
    for (auto& t : threads)
        t.join();
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);

    NESTObservableArray output;
    for (auto& r : chunk_results)
        output += std::move(r);
    return output;
}


void init_array(py::module& m){
    auto m_array = m.def_submodule("array", "array");

    py::class_<NESTObservableArray>(m_array, "NESTObservableArray", py::dynamic_attr())
        .def(py::init<>())
        .def_readonly("s1_nhits", &NESTObservableArray::s1_nhits)
        .def_readonly("s1_nhits_thr", &NESTObservableArray::s1_nhits_thr)
        .def_readonly("s1_nhits_dpe", &NESTObservableArray::s1_nhits_dpe)
        .def_readonly("s1r_phe", &NESTObservableArray::s1r_phe)
        .def_readonly("s1c_phe", &NESTObservableArray::s1c_phe)
        .def_readonly("s1r_phd", &NESTObservableArray::s1r_phd)
        .def_readonly("s1c_phd", &NESTObservableArray::s1c_phd)
        .def_readonly("s1r_spike", &NESTObservableArray::s1r_spike)
        .def_readonly("s1c_spike", &NESTObservableArray::s1c_spike)
        .def_readonly("s2_Nee", &NESTObservableArray::s2_Nee)
        .def_readonly("s2_Nph", &NESTObservableArray::s2_Nph)
        .def_readonly("s2_nhits", &NESTObservableArray::s2_nhits)
        .def_readonly("s2_nhits_dpe", &NESTObservableArray::s2_nhits_dpe)
        .def_readonly("s2r_phe", &NESTObservableArray::s2r_phe)
        .def_readonly("s2c_phe", &NESTObservableArray::s2c_phe)
        .def_readonly("s2r_phd", &NESTObservableArray::s2r_phd)
        .def_readonly("s2c_phd", &NESTObservableArray::s2c_phd)
        .def_readonly("s1_waveform_time", &NESTObservableArray::s1_waveform_time)
        .def_readonly("s1_waveform_amp", &NESTObservableArray::s1_waveform_amp)
        .def_readonly("s2_waveform_time", &NESTObservableArray::s2_waveform_time)
        .def_readonly("s2_waveform_amp", &NESTObservableArray::s2_waveform_amp)
        .def_readonly("n_electrons", &NESTObservableArray::n_electrons)
        .def_readonly("n_photons", &NESTObservableArray::n_photons)
        .def_readonly("s1_photon_times", &NESTObservableArray::s1_photon_times);


    m_array.def("runNESTvec", &runNESTvec,
        "Generate (S1, S2) for a vector of recoil energies",
        py::arg("detector"),
        py::arg("interaction_type"),
        py::arg("energies"),
        py::arg("positions"),
        py::arg("inField") = -1.0,
        py::arg("seed") = 0,
        py::arg("er_yield_params") = default_ERYieldsParam,
        py::arg("nr_yield_params") = default_NRYieldsParam,
        py::arg("width_params") = default_NRERWidthsParam,
        py::arg("s1_mode") = NEST::S1CalculationMode::Hybrid,
        py::arg("s2_mode") = NEST::S2CalculationMode::Full,
        py::arg("calculate_times") = false
    );

    m_array.def("runNESTvec_parallel", &runNESTvec_parallel,
        "Generate (S1, S2) for a vector of recoil energies using multiple threads.\n"
        "Results depend on seed and chunk_size, but not on n_threads (<= 0 uses all cores).",
        py::arg("detector"),
        py::arg("interaction_type"),
        py::arg("energies"),
        py::arg("positions"),
        py::arg("inField") = -1.0,
        py::arg("seed") = 0,
        py::arg("er_yield_params") = default_ERYieldsParam,
        py::arg("nr_yield_params") = default_NRYieldsParam,
        py::arg("width_params") = default_NRERWidthsParam,
        py::arg("s1_mode") = NEST::S1CalculationMode::Hybrid,
        py::arg("s2_mode") = NEST::S2CalculationMode::Full,
        py::arg("calculate_times") = false,
        py::arg("n_threads") = 0,
        py::arg("chunk_size") = 1000,
        py::call_guard<py::gil_scoped_release>()
    );
}