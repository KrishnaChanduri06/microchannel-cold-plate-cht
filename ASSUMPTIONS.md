# Modelling choices and assumptions

## Benchmark data
- Figures 4 to 7 of Qu and Mudawar (2002) were digitised automatically (`digitise/`). Calibration uses the plot frame (axis limits) and is checked against the interior major ticks, all within 0.1 % of their nominal values.
- Figures 4a, 5 and 6 come from the same test runs; their Re values agree point by point within 3 (0.2 %). Digitised Re ranges: 139 to 1674 (paper 139 to 1672) and 386 to 1290 (paper 385 to 1289).
- Repeat runs appear as stacked markers in some Re columns (Re ≈ 140 in Fig. 5; Re ≈ 440 and 955 in Fig. 6a; Re ≈ 787 in Fig. 6b). Where both markers can be measured they are averaged (`n_markers` column).
- Fig. 6a, Re ≈ 953: an overlapping TC4 marker is reassigned from TC1 to TC4; the hidden marker beneath it is not used.
- Fig. 4b (overlapping open and filled markers) uses template matching against the legend symbols.
- Fig. 7 numerical curves are traced from the image; gaps under the in-plot labels are linearly interpolated.
- Fig. 6 shows T<sub>th</sub> − T<sub>in</sub>; Fig. 7 shows absolute T<sub>th</sub>. At Re = 226 the two figures differ by about 2 K, within digitising tolerance.

## Properties
- Water properties from CoolProp. Above about 100 °C, liquid properties are evaluated just above the saturation pressure (pressurised loop).
- Re uses inlet properties (15 °C), as in the paper. ΔP and h in the 1-D model use mean bulk temperature properties. Viscosity drops about 30 % between 15 and 30 °C (1.14 to 0.80 mPa·s).
- CFD: water μ, k and c<sub>p</sub> as polynomials in T (fit error below 0.06 %), constant density at the inlet temperature so the mass flow matches the experiment.
- Copper k = 401 W/(m·K), polycarbonate k = 0.2 W/(m·K), air in the slots k = 0.0263 W/(m·K), all constant.

## Geometry and heat input
- Three air slots, 1.6 mm wide (Sec. 2.2), centred midway between the thermocouples, consistent with the dips in the paper's Fig. 7 curves. Slot height 3.175 mm (level with the thermocouple plane), taken from Fig. 3 as the text does not state it.
- Total heat Q = q″<sub>eff</sub> · A<sub>t</sub> with A<sub>t</sub> = 1.0 × 4.48 cm<sup>2</sup> (Eq. 1): 448 W and 896 W.
- Heater flux applied to the copper part of the unit-cell bottom only (A<sub>bot</sub>, Eq. 14): q″<sub>bot</sub> = 114.3 W/cm<sup>2</sup> at q″<sub>eff</sub> = 100 W/cm<sup>2</sup>. The slots end at the thermocouple plane, so the 1-D thermocouple prediction uses the full width N·W·L.
- Cover top: h = 10 W/(m<sup>2</sup>·K) to 25 °C (Eq. 15). The loss is about 0.03 W, so it is neglected in the 1-D model and kept in the CFD.

## Hydraulics
- Fully developed f·Re (Fanning) and the Hagenbach factor K(∞) from the Shah and London polynomials. f·Re checked with the 2-D solver in `analytical/nusselt_fd.py`: 17.20 vs 17.21 at α = 0.324.
- Plenum losses (Eqs. 19 to 22): no plenum dimensions are given, only K ≈ 1. With plenum velocities much smaller than the channel velocity the kinetic terms cancel between inlet and outlet, leaving ΔP<sub>minor</sub> = (K<sub>c2</sub> + K<sub>e2</sub>) ρu<sup>2</sup>/2, i.e. two velocity heads (upper estimate).
- Wall-viscosity variant of the 1-D model: f/f<sub>iso</sub> = (μ<sub>w</sub>/μ<sub>b</sub>)<sup>0.58</sup> for laminar liquid heating (Deissler's result as given in Kakaç and Yener, *Convective Heat Transfer*, 2nd ed., 1994, Eq. 9.34b; 0.50 for cooling; derived for fully developed flow in circular tubes, so approximate for developing flow in rectangular channels), with the mean wall temperature T<sub>b,mean</sub> + Q·R<sub>conv</sub>.

## Heat transfer (1-D model)
- Nu = 5.30: fully developed H1 condition, three heated walls (bottom and both sides), cover adiabatic, computed with the 2-D finite-volume solver. The solver reproduces the Shah and London four-wall values within 0.002 (3.608 at α = 1, 4.124 at 0.5, 5.333 at 0.25, 4.848 at 0.324) and is grid independent (5.305 to 5.302).
- Nu is based on the heated perimeter, consistent with the convective area N·(W<sub>ch</sub> + 2η<sub>f</sub>H<sub>ch</sub>)·L. The H1 condition is reasonable since η<sub>f</sub> ≈ 0.97.
- Fin efficiency η<sub>f</sub> = tanh(mH<sub>ch</sub>)/(mH<sub>ch</sub>), m = (2h/(k<sub>s</sub>·2W<sub>w</sub>))<sup>1/2</sup>, adiabatic tip (fin in contact with the cover).
- Thermocouple temperature T<sub>tc</sub>(x) = T<sub>f</sub>(x) + Q·[(H<sub>w2</sub> − H<sub>th</sub>)/(k<sub>s</sub>·N·W·L) + R<sub>conv</sub>], with T<sub>f</sub>(x) linear in x.
- Limitation: the fully developed Nu ignores thermal development. L<sub>th</sub> = 0.05 Re Pr D<sub>h</sub> is 18 mm at Re = 139 and 234 mm at Re = 1672 (5.2 L), so the model over-predicts temperatures at high Re.

## Design study
- Product-type plate: 1 cm × 4.48 cm footprint, 1.0 mm base, 1 mm lid of the same solid as the plate (four heated walls), no air slots, 713 µm channel depth, N = 1 cm / (W<sub>ch</sub> + W<sub>fin</sub>) channels.
- Water at 30 °C inlet (warm-water data-centre cooling), uniform 100 W/cm<sup>2</sup> over the footprint; selected Pareto designs repeated at 200 W/cm<sup>2</sup>.
- Silicon k = 148 W/(m·K).
- Cases with inlet Re above about 1800 are near laminar transition and are flagged in the results.

## Icepak model
- The full 21-channel plate requires about 3.1 million cells to resolve the flow (cells of 0.03 mm across and through the channels); this exceeds the student licence limit. A 7-channel strip with the same pitch and edge walls is solved instead, with a 2 mm cover (the cover carries a negligible share of the heat) and constant water properties at the mean bulk temperature.

## Limitations

**Validation model (Fluent unit cell)**
1. One channel with symmetry sides: all 21 channels are assumed identical. The Icepak strip supports this for the base temperature (centre and edge within 0.01 K), but the edge-channel water runs warmer.
2. Slot height (3.175 mm) is read from Fig. 3; the paper does not state it.
3. Inlet and outlet plenums are not modelled. CFD ΔP is channel-only, and the measured ΔP includes plenum losses that cannot be computed without plenum dimensions.
4. Uniform inlet velocity and a zero-gauge outlet; flow maldistribution between channels is not represented.
5. Water density constant; μ, k and c<sub>p</sub> temperature-dependent. Solid properties constant.
6. Laminar, steady, no radiation, no buoyancy. Heat loss is limited to the cover top.
7. Thermocouples are modelled as the y-averaged temperature on the thermocouple plane, not as beads in drilled holes. The CFD reads 5 to 12 % above the measurements, increasing with Re; thermocouple contact resistance and lead conduction are possible causes.
8. Discretisation uncertainty on the 90 000-cell mesh: below 4.6 % for temperatures and about 2 % for ΔP (GCI).

**Benchmark data**
9. Digitised from published figures: about 0.2 % of axis span, about 0.3 K where markers overlap. Repeat-run markers are averaged.

**1-D model**
10. Fully developed Nu and f·Re; no developing-flow correlation. Over-predicts R<sub>th</sub> by up to 62 % when L<sub>th</sub>/L > 3.
11. The (μ<sub>w</sub>/μ<sub>b</sub>)<sup>0.58</sup> correction is for fully developed circular-tube flow and is approximate here.

**Design study**
12. Cold-plate resistance only: chip, thermal interface material and package are not modelled, so junction temperature is not predicted.
13. Uniform heat flux; real processors have hotspots.
14. Straight parallel channels only; manifold and other layouts not studied. Channel depth fixed at 713 µm.
15. 1 mm base and lid, 30 °C water inlet, constant silicon k = 148 W/(m·K) (it falls with temperature in reality).
16. Four high-flow cases (Re 1860 to 2300) are near laminar transition; none is on the Pareto front.
17. Pumping power is for the channels only; headers, manifolds and tubing are excluded.

**Icepak**
18. Full 21-channel plate not solvable within the student licence cell limit (3.1 M cells needed); 7-channel strip solved instead.
19. Icepak temperatures 7 to 15 K above the Fluent model, mainly from the uniform cut-cell mesh (0.25 mm axial cells, no wall clustering) under-resolving the entrance region; constant water properties.
