(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i l g e b c k a d)
(:init 
(harmony)
(planet i)
(planet l)
(planet g)
(planet e)
(planet b)
(planet c)
(planet k)
(planet a)
(planet d)
(province i)
(province l)
(province g)
(province e)
(province b)
(province c)
(province k)
(province a)
(province d)
)
(:goal
(and
(craves i l)
(craves l g)
(craves g e)
(craves e b)
(craves b c)
(craves c k)
(craves k a)
(craves a d)
)))