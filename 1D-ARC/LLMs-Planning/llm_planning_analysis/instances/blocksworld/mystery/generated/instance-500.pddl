(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a e g h l c)
(:init 
(harmony)
(planet a)
(planet e)
(planet g)
(planet h)
(planet l)
(planet c)
(province a)
(province e)
(province g)
(province h)
(province l)
(province c)
)
(:goal
(and
(craves a e)
(craves e g)
(craves g h)
(craves h l)
(craves l c)
)))