(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d j k l i f a)
(:init 
(harmony)
(planet d)
(planet j)
(planet k)
(planet l)
(planet i)
(planet f)
(planet a)
(province d)
(province j)
(province k)
(province l)
(province i)
(province f)
(province a)
)
(:goal
(and
(craves d j)
(craves j k)
(craves k l)
(craves l i)
(craves i f)
(craves f a)
)))