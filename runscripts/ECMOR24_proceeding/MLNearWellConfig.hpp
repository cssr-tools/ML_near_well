// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2025 NORCE Research AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MLNEARWELL_CONFIG_HPP
#define MLNEARWELL_CONFIG_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <cctype>
#include <stdexcept>
#include <vector>

#include <opm/material/common/MathToolbox.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

namespace Opm {

/*!
 * \brief Represents scaling information for a feature.
 *
 * Supports standard (mean/std) and min-max scaling.
 */
struct ScalerValue
{
    enum class Type { None, Standard, MinMax } type = Type::None;
    double mean = 0.0;
    double std  = 1.0;
    double min  = 0.0;
    double max  = 1.0;
    double range_min = -1.0;
    double range_max = 1.0;

    template<class Value>
    Value scale(Value raw_value) const
    {
        switch (type) {
            case Type::Standard:
                return (std == 0.0) ? mean : (raw_value - mean) / std;
            case Type::MinMax: {
                double denom = max - min;
                Value X_std = (denom == 0.0) ? min : (raw_value - min) / denom;
                return X_std * (range_max - range_min) + range_min;
            }
            case Type::None:
            default:
                return raw_value;
        }
    }

    template<class Value>
    Value unscale(Value scaled_value) const
    {
        switch (type) {
            case Type::Standard: return scaled_value * std + mean;
            case Type::MinMax:{
                Value X_std = (scaled_value - range_min) / (range_max - range_min);
                return X_std * (max - min) + min;
            }
            case Type::None:
            default:             return scaled_value;
        }
    }
};

/*!
 * \brief Represents a transformation applied to a feature.
 *
 * Supports log, log10, log1p, and no transform. Provides forward
 * and inverse methods.
 */
struct TransformValue
{
    enum class Type { None, Log10 } type = Type::None;

    TransformValue() = default;
    explicit TransformValue(const std::string& name)
    {
        if      (name == "log10") type = Type::Log10;
        else                       type = Type::None;
    }

    template<class Value>
    Value apply(Value raw_value) const
    {
        switch (type) {
            case Type::Log10: return log10(raw_value);
            case Type::None:
            default:          return raw_value;
        }
    }

    template<class Value>
    Value applyInverse(Value transformed_value) const
    {
        switch (type) {
            case Type::Log10: return pow(10.0, transformed_value);
            case Type::None:
            default:          return transformed_value;
        }
    }
};

/*!
 * \brief Metadata for a single MLNearWell feature (input or output).
 *
 * Includes transformation, scaling, delta flag, and actual feature name.
 */
struct FeatureSpecMLNearWell {
    TransformValue transform;
    ScalerValue scaler;
    std::string actual_name;

    FeatureSpecMLNearWell() = default;
};

/*!
 * \brief Configuration for a Hybrid Newton ML model.
 *
 * Encapsulates model path, cell indices, apply times, input/output
 * features, and validation. Can be constructed from a PropertyTree.
 */
class MLNearWellConfig {
public:
    std::string model_path;
    std::string model_type;
    bool debug;
    std::vector<std::pair<std::string, FeatureSpecMLNearWell>> input_features;
    std::vector<std::pair<std::string, FeatureSpecMLNearWell>> output_features;

    // Model dependent parameters:
    // - for the CO2 models:
    double injection_rate_per_day; // Injection rate in m^3/day

    // - for the CO2 3D model:
    int stencil_size;

    // - for the CO2 3D time model:
    int time_window;
    int first_injection_length;
    int first_break_length;
    int second_injection_length;

    // Default constructor
    MLNearWellConfig() = default;

    /*!
    * \brief Construct configuration from a PropertyTree.
    *
    * Loads model path, cell indices, apply times, and features from
    * the provided PropertyTree. Throws on missing or invalid entries.
    */
    explicit MLNearWellConfig(const PropertyTree& model_config)
    {
        model_path = model_config.get<std::string>("model_path", "");
        model_type = model_config.get<std::string>("model_type", "");
        if (model_path.empty() != model_type.empty()) {
            throw std::runtime_error("Only one of  'model_path' or 'model_type' is specified in MLNearWell config");
        }
        debug = model_config.get<bool>("debug", false);

        // Parse features
        parseFeatures(model_config, "features.inputs", input_features);
        parseFeatures(model_config, "features.outputs", output_features);

        injection_rate_per_day = model_config.get<double>("injection_rate_per_day", 0.0);

        stencil_size = model_config.get<int>("stencil_size", 0);

        time_window = model_config.get<int>("time_window", 0);
        first_injection_length = model_config.get<int>("first_injection_length", 0);
        first_break_length = model_config.get<int>("first_break_length", 0);
        second_injection_length = model_config.get<int>("second_injection_length", 0);
    }

    bool hasInputFeature(const std::string& name) const {
        const std::string name_l = toLowerStr(name);
        return std::any_of(input_features.begin(), input_features.end(),
                           [&](const auto& p) { return toLowerStr(p.first) == name_l; });
    }

    bool hasOutputFeature(const std::string& name) const {
        const std::string name_l = toLowerStr(name);
        return std::any_of(output_features.begin(), output_features.end(),
                           [&](const auto& p) { return toLowerStr(p.first) == name_l; });
    }

    const FeatureSpecMLNearWell& requireInputFeature(const std::string& name) const {
        return requireFeature(input_features, name, "input");
    }

    const FeatureSpecMLNearWell& requireOutputFeature(const std::string& name) const {
        return requireFeature(output_features, name, "output");
    }

    template<class Value>
    Value transformAndScaleInput(const std::string& name, const Value& raw_value) const
    {
        const auto& spec = requireInputFeature(name);
        return spec.scaler.scale<Value>(spec.transform.apply<Value>(raw_value));
    }

    template<class Value>
    Value unscaleAndInverseOutput(const std::string& name, const Value& model_value) const
    {
        const auto& spec = requireOutputFeature(name);
        return spec.transform.applyInverse<Value>(spec.scaler.unscale<Value>(model_value));
    }

    /*!
    * \brief Validate feature compatibility with simulator settings.
    *
    */
    void validateConfig() const {
        if (model_type == "h2o") {
            requireFeature(input_features, "PRESSURE", "input");
            requireFeature(input_features, "ANALYTICAL_PI", "input");
        }
        else if (model_type == "co2_2d") {
            requireFeature(input_features, "PRESSURE", "input");
            requireFeature(input_features, "ANALYTICAL_PI", "input");
            requireFeature(output_features, "WI", "output");

            // NOTE: The log10 transform takes place in co2_2d/upscale.py
            const auto& analytical_pi_spec = requireFeature(input_features, "ANALYTICAL_PI", "input");
            const auto& wi_spec = requireFeature(output_features, "WI", "output");
            if (analytical_pi_spec.transform.type != TransformValue::Type::Log10 || wi_spec.transform.type != TransformValue::Type::Log10) {
                throw std::runtime_error("CO2 2D model was trained with log10 transform for 'ANALYTICAL_PI' and 'WI', but config specifies transforms that are not log10");
            }

        }
        else if (model_type == "co2_3d") {
            if (stencil_size % 2 == 0 or stencil_size <= 0) {
                throw std::runtime_error("Stencil size must be positive and odd for CO2 3D model in MLNearWell config");
            }

            // NOTE: The log10 transform takes place in co2_3d/nn.py
            const auto& analytical_pi_spec = requireFeature(input_features, "ANALYTICAL_PI", "input");
            const auto& wi_spec = requireFeature(output_features, "WI", "output");
            if (analytical_pi_spec.transform.type != TransformValue::Type::Log10 || wi_spec.transform.type != TransformValue::Type::Log10) {
                throw std::runtime_error("CO2 3D model was trained with log10 transform for 'ANALYTICAL_PI' and 'WI', but config specifies transforms that are not log10");
            }
        }
        else if (model_type == "co2_3d_time_in_3d_time") {
            if ((time_window <= 0) or (first_injection_length <= 0) or (first_break_length <= 0)) {
                throw std::runtime_error("Invalid 'time_window' or related parameters for CO2 3D time model in MLNearWell config");
            }

            // NOTE: The log10 transform takes place in co2_3d/nn.py
            const auto& analytical_pi_spec = requireFeature(input_features, "ANALYTICAL_PI", "input");
            const auto& wi_spec = requireFeature(output_features, "WI", "output");
            if (analytical_pi_spec.transform.type != TransformValue::Type::Log10 || wi_spec.transform.type != TransformValue::Type::Log10) {
                throw std::runtime_error("CO2 3D model was trained with log10 transform for 'ANALYTICAL_PI' and 'WI', but config specifies transforms that are not log10");
            }

        }
    }

    /*!
    * \brief Case-insensitive helper: convert string to lowercase (safe for signed 
    * char.
    *
    * \param s        string to convert.
    */

    static std::string toLowerStr(const std::string& s) {
        std::string r; r.reserve(s.size());
        for (unsigned char c : s) r.push_back(static_cast<char>(std::tolower(c)));
        return r;
    }

private:
    /*! 
    * \brief Parse feature specifications from a PropertyTree.
    *
    * Reads transform, scaling parameters, and delta flag.
    * Stores the results in the given `features` vector.
    *
    * \param pt       PropertyTree containing the model configuration.
    * \param path     Path to the feature subtree ("features.inputs" or "features.outputs").
    * \param features Destination vector of (name, FeatureSpec) pairs.
    */
    void parseFeatures(const PropertyTree& pt, const std::string& path,
                       std::vector<std::pair<std::string, FeatureSpecMLNearWell>>& features) {
        auto subtreeOpt = pt.get_child_optional(path);
        if (!subtreeOpt) return;

        for (const auto& name : subtreeOpt->get_child_keys()) {
            const PropertyTree& ft = subtreeOpt->get_child(name);
            FeatureSpecMLNearWell spec;
            spec.transform = TransformValue(ft.get<std::string>("feature_engineering", "none"));

            if (auto sOpt = ft.get_child_optional("scaling_params")) {
                const PropertyTree& s = *sOpt;
                if (s.get_child_optional("mean") && s.get_child_optional("std")) {
                    spec.scaler.type = ScalerValue::Type::Standard;
                    spec.scaler.mean = s.get<double>("mean", 0.0);
                    spec.scaler.std  = s.get<double>("std", 1.0);
                }
                else if (s.get_child_optional("min") && s.get_child_optional("max")) {
                    spec.scaler.type = ScalerValue::Type::MinMax;
                    spec.scaler.min  = s.get<double>("min", 0.0);
                    spec.scaler.max  = s.get<double>("max", 1.0);
                }
                else {
                    spec.scaler.type = ScalerValue::Type::None;
                }
            }
            else {
                spec.scaler.type = ScalerValue::Type::None;
            }

            spec.actual_name = name;

            features.emplace_back(name, std::move(spec));
        }
    }

    const FeatureSpecMLNearWell& requireFeature(const std::vector<std::pair<std::string, FeatureSpecMLNearWell>>& features,
                                      const std::string& name,
                                    const char* feature_kind) const
    {
        const std::string name_l = toLowerStr(name);
        for (const auto& kv : features) {
            if (toLowerStr(kv.first) == name_l) return kv.second;
        }
        throw std::runtime_error(std::string("Missing required ") + feature_kind +
        " feature in MLNearWell config: '" + name + "'");
    }

};

} // namespace Opm

#endif
