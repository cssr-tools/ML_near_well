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

#include <config.h>
#include <opm/simulators/wells/MLNearWellConfig.hpp>

#include <opm/common/ErrorMacros.hpp>

#include <opm/simulators/linalg/PropertyTree.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace Opm {

MLNearWellConfig::MLNearWellConfig(const PropertyTree& model_config)
{
    model_path = model_config.get<std::string>("model_path", "");
    if (model_path.empty()) {
        throw std::runtime_error("Missing 'model_path' in MLNearWell config");
    }
    debug = model_config.get<bool>("debug", false);

    // Parse features
    parseFeatures(model_config, "features.inputs", input_features);
    parseFeatures(model_config, "features.outputs", output_features);

}

bool MLNearWellConfig::hasInputFeature(const std::string& name) const
{
    return std::ranges::any_of(input_features,
                               [&name](const auto& p) { return p.first == name; });
}

bool MLNearWellConfig::hasOutputFeature(const std::string& name) const
{
    return std::ranges::any_of(output_features,
                               [&name](const auto& p) { return p.first == name; });
}

const FeatureSpecMLNearWell& MLNearWellConfig::requireInputFeature(const std::string& name) const
{
    return requireFeature(input_features, name, "input");
}

const FeatureSpecMLNearWell& MLNearWellConfig::requireOutputFeature(const std::string& name) const
{
    return requireFeature(output_features, name, "output");
}

// No specific validation rules implemented yet, but this is where they would go.
void MLNearWellConfig::validateConfig() const {}

void MLNearWellConfig::
parseFeatures(const PropertyTree& pt, const std::string& path,
              std::vector<std::pair<std::string, FeatureSpecMLNearWell>>& features)
{
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

const FeatureSpecMLNearWell& MLNearWellConfig::requireFeature(
    const std::vector<std::pair<std::string, FeatureSpecMLNearWell>>& features,
    const std::string& name,
    const char* feature_kind) const
{
    const auto it = std::ranges::find_if(features,
                                         [&name](const auto& p) { return p.first == name; });
    if (it == features.end()) {
        throw std::runtime_error("Missing required " + std::string(feature_kind) +
                                 " feature in MLNearWell config: '" + name + "'");
    }

    return it->second;
}

} // namespace Opm