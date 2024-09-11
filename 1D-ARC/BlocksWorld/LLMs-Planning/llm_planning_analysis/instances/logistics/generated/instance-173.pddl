(define (problem LG-generalization)
(:domain logistics-strips)(:objects c1 t1 a1 l1-5 p2 l1-3 p3 l1-1 l1-4 l1-0 l1-2 c0 t0 a0 l0-5 p0 l0-3 p1 l0-1 l0-4 l0-0 l0-2)
(:init 
(CITY c1)
(TRUCK t1)
(AIRPLANE a1)
(LOCATION l1-5)
(in-city l1-5 c1)
(OBJ p2)
(at p2 l1-5)
(at t1 l1-5)
(LOCATION l1-3)
(in-city l1-3 c1)
(OBJ p3)
(at p3 l1-3)
(LOCATION l1-1)
(in-city l1-1 c1)
(LOCATION l1-4)
(in-city l1-4 c1)
(LOCATION l1-0)
(in-city l1-0 c1)
(LOCATION l1-2)
(in-city l1-2 c1)
(CITY c0)
(TRUCK t0)
(AIRPLANE a0)
(LOCATION l0-5)
(in-city l0-5 c0)
(OBJ p0)
(at p0 l0-5)
(at t0 l0-5)
(LOCATION l0-3)
(in-city l0-3 c0)
(OBJ p1)
(at p1 l0-3)
(LOCATION l0-1)
(in-city l0-1 c0)
(LOCATION l0-4)
(in-city l0-4 c0)
(LOCATION l0-0)
(in-city l0-0 c0)
(LOCATION l0-2)
(in-city l0-2 c0)
(AIRPORT l1-2)
(at a1 l1-2)
(AIRPORT l0-2)
(at a0 l0-2)
)
(:goal
(and
(at p2 l1-3)
(at p0 l0-3)
(at p3 l0-2)
(at p1 l1-2)
)))