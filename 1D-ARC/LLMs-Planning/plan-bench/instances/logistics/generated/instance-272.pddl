(define (problem LG-generalization)
(:domain logistics-strips)(:objects c0 t0 a0 l0-3 p1 l0-2 p0 l0-0 l0-1)
(:init 
(CITY c0)
(TRUCK t0)
(AIRPLANE a0)
(LOCATION l0-3)
(in-city l0-3 c0)
(OBJ p1)
(at p1 l0-3)
(at t0 l0-3)
(LOCATION l0-2)
(in-city l0-2 c0)
(OBJ p0)
(at p0 l0-2)
(LOCATION l0-0)
(in-city l0-0 c0)
(LOCATION l0-1)
(in-city l0-1 c0)
(AIRPORT l0-1)
(at a0 l0-1)
)
(:goal
(and
(at p1 l0-2)
)))