

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b c)
(craves c a)
(planet d)
(province b)
(province d)
)
(:goal
(and
(craves b c)
(craves d b))
)
)


