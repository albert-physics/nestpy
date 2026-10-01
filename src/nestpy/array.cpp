#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "NEST.hh"
#include "execNEST.hh"

namespace py = pybind11;
using namespace pybind11::literals;


// numpy arrays of doubles, converted (e.g. from lists) only when necessary
using double_array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Read-only numpy view of a field, sharing its memory and keeping owner alive
template <typename T>
static py::array_t<T> field_view(const std::vector<T>& field, py::handle owner){
    py::array_t<T> view(field.size(), field.data(), owner);
    view.attr("setflags")("write"_a = false);
    return view;
}

// Variable-length field as an awkward array with one list per event (copied)
template <typename T>
static py::object ragged_field(const std::vector<std::vector<T>>& field){
    py::array_t<int64_t> counts(field.size());
    size_t total = 0;
    for (size_t i = 0; i < field.size(); ++i){
        counts.mutable_at(i) = static_cast<int64_t>(field[i].size());
        total += field[i].size();
    }
    py::array_t<T> content(total);
    T* out = content.mutable_data();
    for (const auto& event : field)
        out = std::copy(event.begin(), event.end(), out);
    return py::module_::import("awkward").attr("unflatten")(content, counts);
}

// Getter for a field: one value per event gives a numpy view, a list per event
// gives an awkward array
template <typename T>
static auto field_getter(std::vector<T> NESTObservableArray::*field){
    return [field](py::object self){
        return field_view(self.cast<const NESTObservableArray&>().*field, self);
    };
}

template <typename T>
static auto field_getter(std::vector<std::vector<T>> NESTObservableArray::*field){
    return [field](const NESTObservableArray& self){ return ragged_field(self.*field); };
}

// Awkward array with one record per event, built from the result's fields.
// The result is passed on the heap: NESTObservableArray has no move constructor,
// so passing it by value would copy every field.
static py::object to_awkward(std::unique_ptr<NESTObservableArray> output,
                             const std::vector<std::string>& field_names){
    // The Python object owns the memory that the numpy fields point into
    py::object result = py::cast(output.release(), py::return_value_policy::take_ownership);
    py::dict columns;
    for (const auto& name : field_names)
        columns[name.c_str()] = result.attr(name.c_str());
    return py::module_::import("awkward").attr("Array")(columns);
}

// Check energies has shape (n,) and positions shape (n, 3), and return n
static size_t check_inputs(const double_array& energies, const double_array& positions){
    const size_t n_events = energies.size();
    if (energies.ndim() != 1)
        throw std::invalid_argument("energies must be 1-dimensional");
    if (n_events > 0 and (positions.ndim() != 2 or positions.shape(1) != 3))
        throw std::invalid_argument("positions must have shape (n_events, 3)");
    if (positions.size() != 3 * n_events)
        throw std::invalid_argument("energies and positions must have the same length");
    return n_events;
}

// Events [begin, end) in the form runNESTvec takes
static std::vector<double> energy_list(const double* energies, size_t begin, size_t end){
    return std::vector<double>(energies + begin, energies + end);
}

static std::vector<std::vector<double>> position_list(const double* positions, size_t begin, size_t end){
    std::vector<std::vector<double>> pos3dxyz;
    pos3dxyz.reserve(end - begin);
    for (size_t i = begin; i < end; ++i)
        pos3dxyz.push_back({positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]});
    return pos3dxyz;
}

// Seed for one chunk. Hashing the user seed first stops nearby seeds sharing
// chunk seeds (with seed + chunk, seed 42 chunk 1 would equal seed 43 chunk 0)
static uint64_t chunk_seed(uint64_t seed, uint64_t chunk){
    return splitmix64(seed) + chunk;
}

// Multithreaded runNESTvec. Events are split into fixed-size chunks, each run
// by runNESTvec with its own seed, so the output depends on the seed and
// chunk_size but not on the number of threads. energies has n_events values
// and positions holds n_events (x, y, z) triples, row-major. The results are
// written into output.
static void runNESTvec_parallel(
    NESTObservableArray& output,
    VDetector* detector,
    INTERACTION_TYPE particleType,
    const double* energies,
    const double* positions,
    size_t n_events,
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
    if (chunk_size <= 0)
        throw std::invalid_argument("chunk_size must be positive");

    const size_t n_chunks = (n_events + chunk_size - 1) / chunk_size;
    if (n_threads <= 0)
        n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = static_cast<int>(std::min<size_t>(n_threads, std::max<size_t>(n_chunks, 1)));

    // Configure the shared detector once, so worker threads only ever read it
    NESTcalc(detector).SetDensity(detector->get_T_Kelvin(), detector->get_p_bar());

    // Each chunk moves its results straight into its slice of the output
    output.resize(n_events);

    std::vector<std::exception_ptr> errors(n_threads);
    std::atomic<size_t> next_chunk{0};

    auto worker = [&](int thread_id){
        try {
            for (size_t c = next_chunk++; c < n_chunks; c = next_chunk++){
                const size_t begin = c * chunk_size;
                const size_t end = std::min(begin + chunk_size, n_events);
                output.assign_events(begin, runNESTvec(
                    detector, particleType,
                    energy_list(energies, begin, end), position_list(positions, begin, end),
                    inField, chunk_seed(seed, c),
                    ERYieldsParam, NRYieldsParam, NRERWidthsParam,
                    s1mode, s2mode, calculate_times));
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
}


void init_array(py::module& m){
    auto m_array = m.def_submodule("array", "array");

    py::class_<NESTObservableArray> observables(m_array, "NESTObservableArray", py::dynamic_attr());
    observables.def(py::init<>());

    // Each field becomes a read-only attribute and a field of the awkward output
    std::vector<std::string> field_names;
    auto add_field = [&](const char* name, auto member){
        observables.def_property_readonly(name, field_getter(member));
        field_names.push_back(name);
    };
    add_field("s1_nhits", &NESTObservableArray::s1_nhits);
    add_field("s1_nhits_thr", &NESTObservableArray::s1_nhits_thr);
    add_field("s1_nhits_dpe", &NESTObservableArray::s1_nhits_dpe);
    add_field("s1r_phe", &NESTObservableArray::s1r_phe);
    add_field("s1c_phe", &NESTObservableArray::s1c_phe);
    add_field("s1r_phd", &NESTObservableArray::s1r_phd);
    add_field("s1c_phd", &NESTObservableArray::s1c_phd);
    add_field("s1r_spike", &NESTObservableArray::s1r_spike);
    add_field("s1c_spike", &NESTObservableArray::s1c_spike);
    add_field("s2_Nee", &NESTObservableArray::s2_Nee);
    add_field("s2_Nph", &NESTObservableArray::s2_Nph);
    add_field("s2_nhits", &NESTObservableArray::s2_nhits);
    add_field("s2_nhits_dpe", &NESTObservableArray::s2_nhits_dpe);
    add_field("s2r_phe", &NESTObservableArray::s2r_phe);
    add_field("s2c_phe", &NESTObservableArray::s2c_phe);
    add_field("s2r_phd", &NESTObservableArray::s2r_phd);
    add_field("s2c_phd", &NESTObservableArray::s2c_phd);
    add_field("s1_waveform_time", &NESTObservableArray::s1_waveform_time);
    add_field("s1_waveform_amp", &NESTObservableArray::s1_waveform_amp);
    add_field("s2_waveform_time", &NESTObservableArray::s2_waveform_time);
    add_field("s2_waveform_amp", &NESTObservableArray::s2_waveform_amp);
    add_field("n_electrons", &NESTObservableArray::n_electrons);
    add_field("n_photons", &NESTObservableArray::n_photons);
    add_field("s1_photon_times", &NESTObservableArray::s1_photon_times);

    m_array.def("runNESTvec",
        [field_names](VDetector* detector, INTERACTION_TYPE particleType,
                      double_array energies, double_array positions,
                      double inField, uint64_t seed,
                      std::vector<double> ERYieldsParam,
                      std::vector<double> NRYieldsParam,
                      std::vector<double> NRERWidthsParam,
                      S1CalculationMode s1mode, S2CalculationMode s2mode,
                      bool calculate_times){
            const size_t n_events = check_inputs(energies, positions);
            // Constructing from the returned value directly avoids a copy (C++17)
            std::unique_ptr<NESTObservableArray> output(new NESTObservableArray(runNESTvec(
                detector, particleType,
                energy_list(energies.data(), 0, n_events), position_list(positions.data(), 0, n_events),
                inField, seed, std::move(ERYieldsParam), std::move(NRYieldsParam),
                std::move(NRERWidthsParam), s1mode, s2mode, calculate_times)));
            return to_awkward(std::move(output), field_names);
        },
        "Generate (S1, S2) for a vector of recoil energies, returned as an awkward\n"
        "array with one record per event.\n"
        "energies has shape (n,) and positions shape (n, 3); float64 numpy arrays are\n"
        "read directly and other inputs (e.g. lists) are converted.",
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

    m_array.def("runNESTvec_parallel",
        [field_names](VDetector* detector, INTERACTION_TYPE particleType,
                      double_array energies, double_array positions,
                      double inField, uint64_t seed,
                      const std::vector<double>& ERYieldsParam,
                      const std::vector<double>& NRYieldsParam,
                      const std::vector<double>& NRERWidthsParam,
                      S1CalculationMode s1mode, S2CalculationMode s2mode,
                      bool calculate_times, int n_threads, int chunk_size){
            // Validate while holding the GIL; no Python objects are touched once it is released
            const size_t n_events = check_inputs(energies, positions);
            auto output = std::make_unique<NESTObservableArray>();
            {
                py::gil_scoped_release release;
                runNESTvec_parallel(
                    *output, detector, particleType, energies.data(), positions.data(), n_events,
                    inField, seed, ERYieldsParam, NRYieldsParam, NRERWidthsParam,
                    s1mode, s2mode, calculate_times, n_threads, chunk_size);
            }
            return to_awkward(std::move(output), field_names);
        },
        "Multithreaded runNESTvec, returned as an awkward array with one record per event.\n"
        "Events are simulated in chunks of chunk_size, each with its own seed derived from\n"
        "seed, so results depend on seed and chunk_size but not on n_threads (<= 0 uses\n"
        "all cores). They differ from runNESTvec's results for the same seed.",
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
        py::arg("chunk_size") = 1000
    );
}
