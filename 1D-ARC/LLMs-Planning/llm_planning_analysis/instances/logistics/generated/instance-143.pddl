(define (problem LG-generalization)
(:domain logistics-strips)(:objects c0 t0 a0 l0-1 p2 l0-0 p1 l0-2 p0)
(:init 
(CITY c0)
(TRUCK t0)
(AIRPLANE a0)
(LOCATION l0-1)
(in-city l0-1 c0)
(OBJ p2)
(at p2 l0-1)
(at t0 l0-1)
(LOCATION l0-0)
(in-city l0-0 c0)
(OBJ p1)
(at p1 l0-0)
(LOCATION l0-2)
(in-city l0-2 c0)
(OBJ p0)
(at p0 l0-2)
(AIRPORT l0-2)
(at a0 l0-2)
)
(:goal
(and
(at p2 l0-0)
(at p1 l0-2)
)))