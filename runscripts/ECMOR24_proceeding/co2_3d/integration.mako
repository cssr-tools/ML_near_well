"""Set the full path to the flow executable and flags"""
${FLOW} ${USEMLNEARWELL} --MLNearWellConfigFile=${MLNEARWELLCONFIGFILE} --linear-solver-reduction=1e-5 --relaxed-max-pv-fraction=0 --newton-max-iterations=50 --newton-min-iterations=5 --tolerance-mb=1e-7 --tolerance-wells=1e-5 --relaxed-well-flow-tol=1e-5 --use-multisegment-well=false --enable-tuning=true --enable-opm-rst-file=true --linear-solver=cprw --enable-well-operability-check=false --min-time-step-before-shutting-problematic-wells-in-days=1e-1

"""Set the model parameters"""
co2store no_disgas_no_diffusion ${RUN_NAME} #Model (co2store/h2store)
tensor3d 0                      #Grid type (radial/cake/cartesian2d/cartesian/cave) and size (theta[in degrees]/theta[in degrees]/width[m]/anynumber(the y size is set equal to the x one))
${RESERVOIR_SIZE} ${HEIGHT}      #Reservoir dimensions [m] (length and height)
${GRID_SIZE} ${NUM_ZCELLS} 0     #Number of x- and z-cells [-] and exponential factor for the telescopic x-gridding (0 to use an equidistance partition)
${2*WELL_RADIUS} 0 0            #Well diameter [m], well transmiscibility (0 to use the computed one internally in Flow), and remove the smaller cells than the well diameter
${INIT_PRESSURE} ${INIT_TEMPERATURE}  0 #Pressure [Pa] on the top, uniform temperature [°], and initial phase in the reservoir (0 wetting, 1 non-wetting)
1e10 1                          #Pore volume multiplier on the boundary [-] (0 to use well producers instead) and deactivate cross flow within the wellbore (see XFLOW in OPM Manual)
0 5 10                          #Activate perforations [-], number of well perforations [-], and length [m]
${NUM_LAYERS} 0 0                           #Number of layers [-] and hysteresis (1 to activate) and econ for the producer (for h2 models)
0 0 0 0 0 0 0                   #Initial salt concentration [kg/m3], salt solubility limit [kg/m3], and precipitated salt density [kg/m3] (for saltprec)
0                               #The function for the reservoir surface

"""Set the saturation functions"""
krw * ((sw - swi) / (1.0 - sni - swi)) ** nkrw             #Wetting rel perm saturation function [-]
krn * ((1.0 - sw - sni) / (1.0 - sni - swi)) ** nkrn      #Non-wetting rel perm saturation function [-]
pec * ((sw - swi) / (1.0 - sni - swi)) ** (-(1.0 / npe))  #Capillary pressure saturation function [Pa]

"""Properties saturation functions"""
"""swi [-], sni [-], krn [-], krw [-], pec [Pa], nkrw [-], nkrn [-], npe [-], threshold cP evaluation, ignore swi for cP"""
<%
safu_rows = [
    "SWI0 0.32 SNI0 0.1 KRW0 1 KRN0 1 PRE0 6120 NKRW0 2.0 NKRN0 2.0 HNPE0 2.0 THRE0 1e-4 IGN0 0",
    "SWI1 0.14 SNI1 0.1 KRW1 1 KRN1 1 PRE1 6120 NKRW1 2.0 NKRN1 2.0 HNPE1 2.0 THRE1 1e-4 IGN1 0",
    "SWI2 0.12 SNI2 0.1 KRW2 1 KRN2 1 PRE2 6120 NKRW2 2.0 NKRN2 2.0 HNPE2 2.0 THRE2 1e-4 IGN2 0",
    "SWI3 0.12 SNI3 0.1 KRW3 1 KRN3 1 PRE3 6120 NKRW3 2.0 NKRN3 2.0 HNPE3 2.0 THRE3 1e-4 IGN3 0",
    "SWI4 0.10 SNI4 0.1 KRW4 1 KRN4 1 PRE4 6120 NKRW4 2.0 NKRN4 2.0 HNPE4 2.0 THRE4 1e-4 IGN4 0",
]
%>${"\n".join(safu_rows)}

"""Properties rock"""
"""Kxy [mD], Kz [mD], phi [-], thickness [m]"""
<%
perms = [context.kwargs[f"PERM_{i}"] for i in range(NUM_LAYERS)]
rock_rows = [
    f"{i+1} {perm} {0.5*perm} {POROSITY} {HEIGHT/NUM_LAYERS}"
    for i, perm in enumerate(perms)
]
%>${"\n".join(rock_rows)}

"""Define the injection values"""
<%
inj1 = int(round(float(INJ1_DAYS)))
shut = int(round(float(SHUT_DAYS)))
total = int(round(float(INJECTION_TIME))) 

if inj1 + shut > total - 1:
    shut = max(0, total - 1 - inj1)
    if inj1 > total - 1:
        inj1 = total - 1
        shut = 0

inj2_final = total - inj1 - shut
durations = [inj1, shut, inj2_final]

reportstep = float(REPORTSTEP_LENGTH)
%>
% for i, inj_step in enumerate(inj):
${"%.6f" % durations[i]} ${reportstep} ${reportstep} ${inj_step[3]} ${float(inj_step[4]) * float(INJECTION_RATE) / 6}
% endfor