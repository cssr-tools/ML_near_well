"""Set the full path to the flow executable and flags"""
${FLOW} --ml-wi-filename="" --linear-solver-reduction=1e-5 --relaxed-max-pv-fraction=0 --ecl-enable-drift-compensation=0 --newton-max-iterations=50 --newton-min-iterations=5 --tolerance-mb=1e-7 --tolerance-wells=1e-5 --relaxed-well-flow-tol=1e-5 --use-multisegment-well=false --enable-tuning=true --enable-opm-rst-file=true --linear-solver=cprw --enable-well-operability-check=false --min-time-step-before-shutting-problematic-wells-in-days=1e-99

"""Set the model parameters"""
co2store no_disgas_no_diffusion     #Model (co2store/h2store/co2eor/saltprec) and name of the template file (see src/pyopmnearwell/templates/)
cake 60                             #Grid type (radial/cake/cartesian2d/cartesian/cpg3d/coord2d/coord3d/tensor2d/tensor3d) and size (theta[in degrees]/theta[in degrees]/width[m]/anynumber(the y size is set equal to the x one))
${LENGTH} ${INT_HEIGHT}             #Reservoir dimensions [m] (Length and height)
${NUM_XCELLS} 1 4                   #Number of x- and z-cells [-] and exponential factor for the telescopic x-gridding (0 to use an equidistance partition)
${2*WELL_RADIUS} 1 1                #Well diameter [m], well transmiscibility (0 to use the computed one internally in Flow), and remove the smaller cells than the well diameter
${INIT_PRESSURE} ${INIT_TEMPERATURE} 0     #Pressure [Pa] on the top, uniform temperature [°], and initial phase in the reservoir (0 wetting, 1 non-wetting)
1e10 1                              #Pore volume multiplier on the boundary [-] (0 to use well producers instead) and deactivate cross flow within the wellbore (see XFLOW in OPM Manual)
0 5 ${int(HEIGHT)}                   #Activate perforations [-], number of well perforations [-], and length [m]
1 0 0                               #Number of layers [-], hysteresis (Killough, Carlson, or 0 to neglect it), and econ for the producer (for h2 models)
0 0 0 0 0 0 0                       #Ini salt conc [kg/m3], salt sol lim [kg/m3], prec salt den [kg/m3], gamma [-], phi_r [-], npoints [-], and threshold [-]  (all entries for saltprec)
0                                   #The function for the reservoir surface

"""Set the saturation functions"""
krw * ((sw - swi) / (1.0 - sni -swi)) ** nkrw             #Wetting rel perm saturation function [-]
krn * ((1.0 - sw - sni) / (1.0 - sni - swi)) ** nkrn      #Non-wetting rel perm saturation function [-]
pec * ((sw - swi) / (1.0 - swi)) ** (-(1.0 / npe)) #Capillary pressure saturation function [Pa]

"""Properties saturation functions"""
"""swi [-], sni [-], krn [-], krw [-], pec [Pa], nkrw [-], nkrn [-], npe [-], threshold cP evaluation, ignore swi for cP"""
SWI1 0. SNI1 0.0 KRW1 1 KRN1 1 PRE1 3000 NKRW1 2 NKRN1 2 HNPE1 2 THRE1 1e-4 IGN1 0

"""Properties rock"""
"""Kxy [mD], Kz [mD], phi [-], thickness [m]"""
PERMXY1 ${PERM} PERMZ1 ${PERM} PORO1 ${POROSITY} THIC1 ${INT_HEIGHT}

"""Define the injection values""" 
"""injection time [d], time step size to write results [d], maximum time step [d], fluid (0 wetting, 1 non-wetting), injection rates [kg/day]"""
${INJECTION_TIME} ${REPORTSTEP_LENGTH} 1e-1 0 ${INJECTION_RATE/6} # Divided by 6, since the model runs on a cake of 60°
