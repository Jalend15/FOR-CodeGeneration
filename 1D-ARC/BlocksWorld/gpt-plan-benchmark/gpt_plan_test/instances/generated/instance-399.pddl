(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k c g a e j l h f b)
(:init 
(handempty)
(ontable k)
(ontable c)
(ontable g)
(ontable a)
(ontable e)
(ontable j)
(ontable l)
(ontable h)
(ontable f)
(ontable b)
(clear k)
(clear c)
(clear g)
(clear a)
(clear e)
(clear j)
(clear l)
(clear h)
(clear f)
(clear b)
)
(:goal
(and
(on k c)
(on c g)
(on g a)
(on a e)
(on e j)
(on j l)
(on l h)
(on h f)
(on f b)
)))