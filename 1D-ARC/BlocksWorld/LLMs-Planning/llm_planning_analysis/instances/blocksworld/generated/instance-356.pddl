(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c f g l k i h b e a)
(:init 
(handempty)
(ontable c)
(ontable f)
(ontable g)
(ontable l)
(ontable k)
(ontable i)
(ontable h)
(ontable b)
(ontable e)
(ontable a)
(clear c)
(clear f)
(clear g)
(clear l)
(clear k)
(clear i)
(clear h)
(clear b)
(clear e)
(clear a)
)
(:goal
(and
(on c f)
(on f g)
(on g l)
(on l k)
(on k i)
(on i h)
(on h b)
(on b e)
(on e a)
)))