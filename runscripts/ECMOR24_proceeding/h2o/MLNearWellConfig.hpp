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

#ifndef MLNearWell_CONFIG_HPP
#define MLNearWell_CONFIG_HPP

#include <opm/simulators/flow/HybridNewtonConfig.hpp>

#include <opm/material/common/MathToolbox.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace Opm {

class PropertyTree;

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
struct FeatureSpecMLNearWell
{
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
class MLNearWellConfig
{
public:
    std::string model_path;
    bool debug;
    std::vector<std::pair<std::string, FeatureSpecMLNearWell>> input_features;
    std::vector<std::pair<std::string, FeatureSpecMLNearWell>> output_features;

    // Default constructor
    MLNearWellConfig() = default;

    /*!
    * \brief Construct configuration from a PropertyTree.
    *
    * Loads model path, cell indices, apply times, and features from
    * the provided PropertyTree. Throws on missing or invalid entries.
    */
    explicit MLNearWellConfig(const PropertyTree& model_config);

    bool hasInputFeature(const std::string& name) const;

    bool hasOutputFeature(const std::string& name) const;

    const FeatureSpecMLNearWell& requireInputFeature(const std::string& name) const;

    const FeatureSpecMLNearWell& requireOutputFeature(const std::string& name) const;

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
    void validateConfig() const;

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
                       std::vector<std::pair<std::string, FeatureSpecMLNearWell>>& features);

    const FeatureSpecMLNearWell& requireFeature(const std::vector<std::pair<std::string, FeatureSpecMLNearWell>>& features,
                                      const std::string& name,
                                      const char* feature_kind) const;
};

} // namespace Opm

#endif