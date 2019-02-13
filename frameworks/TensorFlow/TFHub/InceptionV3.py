import tensorflow as tf
import tensorflow_hub as hub



tfhub_module_url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
initializer = tf.truncated_normal


def main(_):
    _graph = tf.Graph()
    with _graph.as_default():
        module_spec = hub.load_module_spec(tfhub_module_url)
        tf.logging.info(msg='Loaded module_spec: %s' % module_spec)
        module_spec_tag_set = module_spec.get_tags()[1]
        tf.logging.info(msg='TensorFlow-Hub ModuleSpec: module_spec\'s tags: %s' % module_spec_tag_set)
        module_spec_tag_set_signature_names = module_spec.get_signature_names(module_spec_tag_set)
        tf.logging.info(msg='TensorFlow-Hub ModuleSpec: module_spec\'s signature names for tag set \'%s\': %s' % (module_spec_tag_set, module_spec_tag_set_signature_names))
        module_spec_tag_set_sig_def_input_info_dict = module_spec.get_input_info_dict(signature=module_spec_tag_set_signature_names[1], tags=module_spec_tag_set)
        tf.logging.info(msg='TensorFlow-Hub ModuleSpec: module_spec\'s input info dict for signature \'%s\' and tags %s is: %s' % (module_spec_tag_set_signature_names[1], module_spec_tag_set, module_spec_tag_set_sig_def_input_info_dict))
        height, width = hub.get_expected_image_size(module_spec)
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name='resized_input_tensor')
        m = hub.Module(module_spec, trainable=True)
        bottleneck_tensor = m(resized_input_tensor)
        trainable_vars = tf.trainable_variables()
        # kernel = tf.get_variable(trainable_vars[0].name)
        init = tf.global_variables_initializer()
        # TODO: This fails, because this probably isn't how this is supposed to be done:
        # for trainable_var in trainable_vars:
        #     trainable_var.initializer = tf.initializers.he_normal()

    with tf.Session(graph=_graph) as sess:
        init.run()
        trainable_vars_with_weights = [trainable_var for trainable_var in trainable_vars if 'weights' in trainable_var.name]
        tf.logging.info(msg='trainable_vars[0]: %s' % trainable_vars[0])
        tf.logging.info(msg='trainable_vars[0] initial weights: %s' % trainable_vars[0].eval(sess))
        first_trainable_var_weight_init_op = [sess.graph.get_operations()[6], sess.graph.get_operations()[7], sess.graph.get_operations[8], 'continues...']
        # TODO: What is going on here?
        first_trainable_var_op = sess.graph.get_operation_by_name('module/InceptionV3/Conv2d_1a_3x3/weights')
        print(sess.graph.get_operations()[10])
        print(sess.graph.get_operation_by_name('module/InceptionV3/Conv2d_1a_3x3/weights/Initializer'))
        print(trainable_vars)


tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run()
