(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e c g b f)
(:init 
(handempty)
(ontable e)
(ontable c)
(ontable g)
(ontable b)
(ontable f)
(clear e)
(clear c)
(clear g)
(clear b)
(clear f)
)
(:goal
(and
(on e c)
(on c g)
(on g b)
(on b f)
)))