# Mask R-CNN for Object Detection and Segmentation

```
def save(self,save_path):
        self.keras_model.save(save_path)
```

```
self.keras_model.fit_generator(
           train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        #######using session and saving .pb file##
        frozen_graph = self.freeze_session(K.get_session(),
                              output_names=[out.op.name for out in self.keras_model.outputs])
        tf.train.write_graph(frozen_graph, self.log_dir, "my_model.pb", as_text=False)
```

this will automatically save a .pb file in your logs directory while you are training your model.
you can check both, the .h5 model file and the .pb file using netron, it is a great visualizer for these architectures.
