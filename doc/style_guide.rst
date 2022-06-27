:orphan:

********************************
Coding Style Guide with Examples
********************************

This guide follows mostly:
https://github.com/numpy/numpy/blob/master/doc/example.py.

In the Python code blocks below, some remarks are included using JavaScript
style notation <!-- comment -->. They provide additional information regarding
the code that is being presented as example.

Module docstring
----------------

The module should start with its own docstring in the first line.
The example below illustrates show it should be implemented.

.. code-block:: python

    """
    Here comes the module description

    Detailed introduction and tutorial of method goes here.

    You can even link tutorials here, if appropriate.

    If you have references you can insert them here, otherwise in the corresponding
    functions, methods or classes.

    For a working example see
    :py:mod:`unitary_event_analysis.py<elephant.unitary_event_analysis>`
    """


Function docstring. Naming conventions
--------------------------------------

The function below illustrates how arguments and functions should be named
throughout Elephant.

.. code-block:: python

    def pair_of_signals_example(spiketrain_i, spiketrain_j):
        # Add '_i' and '_j' suffixes to a pair of signals, spiketrains or any
        # other variables that come in pairs.

    def perfect_naming_of_parameters(spiketrains, spiketrain, reference_spiketrain,
                         target_spiketrain, signal, signals, max_iterations,
                         min_threshold, n_bins, n_surrogates, bin_size, max_size,
                         time_limits, time_range, t_start, t_stop, period, order,
                         error, capacity, source_matrix, cov_matrix,
                         selection_method='aic'
                         ):
        r"""
        Full example of the docstring and naming conventions.

        Function names should be in lowercase, with words written in full, and
        separated by underscores. Exceptions are for common abbreviations, such
        as "psd" or "isi". But words such as "coherence" must be written in full,
        and not truncated (e.g. "cohere").

        If the truncation or abbreviation not in conformity to this naming
        convention was adopted to maintain similarity to a function used
        extensively in another language or package, mention this in the "Notes"
        section, like the comment below:
        <!--
        Notes
        -----
        This function is similar to `welch_cohere` function in MATLAB.
        -->

        The rationale for the naming of each parameter in this example will be
        explained in the relevant "Parameters" section. Class parameters and
        attributes also follow the same naming convention.

        Parameters
        ----------
        <!-- As a general rule, each word is written in full lowercase, separated
        by underscores. Special cases apply according to the examples below -->
        spiketrains : neo.SpikeTrain or list of neo.SpikeTrain
            Within Elephant, this is how to name an input parameter that contains
            at least one spike train. The parameter name is in plural (i.e.,
            `spiketrains`). The function will deal with converting a single
            `neo.SpikeTrain` to a list of `neo.SpikeTrain` if needed.
            Note that although these are two words, they are NOT separated by
            underscore because Neo does not use underscore, and Elephant must keep
            compatibility. Do not use names such as `sts`, `spks`, or
            `spike_trains`.
        spiketrain: neo.SpikeTrain
            If the function EXPLICITLY requires only a single spike train, then
            the parameter should be named in singular (i.e., `spiketrain`). Do
            not use names such as `st`, `spk`, or `spike_train`.
        reference_spiketrain : neo.SpikeTrain
            If a function uses more than one parameter with single spike trains,
            then each parameter name begins with a meaningful name,
            followed by "_spiketrain" in singular form.
        target_spiketrain: neo.SpikeTrain
            Second parameter that is a single spike train. Note that the difference
            from `reference_spiketrain` is indicated by a meaningful name at the
            beginning.
        signal : neo.AnalogSignal
            If a single `neo.AnalogSignal` object is passed to the function, even if
            it contains several signals (arrays).
        signals : list of neo.AnalogSignal
            If the parameter is a container that has at least one `neo.AnalogSignal`
            object. The name of the parameter is `signals` (plural).
        max_iterations : int
            Parameters that represent a maximum value should start with "max_"
            prefix, followed by the description as a full word. Therefore, do not
            use names such as `max_iter` or `maxiter`.
        min_threshold : float
            Same case as for maximum. Parameters that represent a minimum value
            should start with "min_" prefix, followed by the description as a full
            word. Therefore, do not use names such as `min_thr` or `minthr`.
        n_bins : int
            Parameters that represent a number should start with the prefix "n_".
            Do not use `numbins`, `bin_number`, or `num_bins`. The prefix should
            be followed by a meaningful word in full.
        n_surrogates : int
            The description should always be meaningful an without abbreviations.
            Therefore, do not use terms as `n` or `n_surr`, that are not
            immediately understood.
        bin_size : pq.Quantity or int
            Separate the words by underscore. Do not use `bin_size`. Old functions
            which use `binsize` are deprecated.
        max_size : float
            Another example showing that words should be separated by underscores.
            This intersects with the naming convention for a maximum value.
        time_limits: list or tuple
            For parameters that define minimum and maximum values as a list or
            tuple (e.g., [-2, 2]), the parameter must start with a meaningful
            word followed by the suffix "_limits". Preferentially, one should use
            two separated parameters (e.g., `max_time` and `min_time` following
            the convention for maximum and minimum already mentioned). But should
            the function require the definition of limits in this form, use the
            name `_limits` and not `_range` (see next parameter).
        time_range: list
            For parameters that behave like a Python range (e.g. [1, 2, 3, 4])), in
            the sense that it is a sequence, not only the lower and upper limits
            as in the example above, the parameter should start with a meaningful
            name followed by the suffix "_range".
        t_start : pq.Quantity
            Standard name within Elephant for defining starting times.
        t_stop : pq.Quantity
            Standard name within Elephant for defining stopping times.
        period : pq.Quantity
            Oscillation period.
            Always use informative names. In this case, one could name the
            parameter as simply as `T`, since this is standard for referring to
            periods. If the function is implementing computations based on a paper
            that has a formula with a variable "T", acknowledge this after
            describing the formula in the docstring. Therefore, write a sentence
            like "`period` refers to :math:`T`"
            If the Elephant function uses an external function (such as from
            `scipy`), and such function has an argument named `T`, also
            acknowledge this in the docstring. Therefore, write a sentence like
            "`period` is forwarded as argument `T` of `scipy.uses_T` function".
            If the external function already has an informative parameter name
            (such as `period`), the same parameter name can be used in the Elephant
            function if forwarded.
            If several input parameters are forwarded or are members
            of a formula, the docstring can present them together as a list.
            But always use informative names, not single letter names if this is
            how they are described in the paper or implemented in another function.
        order : int
            Order of the Butterworth filter.
            This is an example of how the `N` parameter of `scipy.signal.butter`
            function could be provided by the user of the Elephant function.
            The docstring would present a text similar to
            "`order` is passed as the `N` argument for `scipy.signal.butter` function".
            Also, in the code implementation, use keyword arguments to make this
            explicit (see the implementation of the function below)
        error : float
            In the case the function has an input parameter that corresponds to a
            greek letter in a formula (in a paper, for instance) always use the
            meaning of the greek letter. Therefore, should :math:`\epsilon` refer
            to the error in the formula, the parameter should be named `error`. As
            already mentioned, this is acknowledged in the docstring after the
            description of the formula.
        capacity : float
            Capacity value.
            When using parameters based on a paper (which, e.g., derives some
            formula), and the parameter's name in this paper is a single letter
            (such as `C` for capacity), always use the meaning
            of the letter. Therefore, the parameter should be named `capacity`,
            not `C`. Acknowledge this in the docstring as already mentioned.
        source_matrix: np.ndarray
            Parameters that are matrices should end with the suffix "_matrix", and
            start with a meaningful name.
        cov_matrix: np.ndarray
            A few exceptions allow the use of abbreviations instead of full words
            in the name of the parameter. These are:
            * "cov" for "covariance" (e.g., `cov_matrix`)
            * "lfp" for "local_field_potential" (e.g. `lfp_signal`)
            * "corr" for "correlation" (e.g. `corr_matrix`).
            THESE EXCEPTIONS ARE NOT ACCEPTED FOR FUNCTION NAMES. Therefore, a
            parameter would be named `cov_matrix`, but the function would be named
            `calculate_covariance_matrix`. If the function name becomes very long,
            then an alias may be created and described appropriately in the "Notes"
            section, as mentioned above. For aliases, see example below.
        selection_method : {'aic', 'bic'}
            Metric for selecting the autoregressive model.
            If 'aic', uses the Akaike Information Criterion (AIC).
            If 'bic', uses Bayesian Information Criterion (BIC).
            Default: 'bic', because it is more reliable than AIC due to the
            mathematical properties (see Notes [3]).
            <!-- Note that the default value that comes in the last line is
            followed by comma and a brief reasoning for defining the default
            `selection_method`). -->

        <!-- Other remarks:
        1. Do not use general parameter names, such as `data` or `matrix`.
        2. Do not use general result names, such as `result` or `output`.
        3. Avoid capitalization (such as the examples mentioned for parameters
           such as `T` for period, or `C` for capacity or a correlation matrix.
        -->

        Returns
        -------
            frequency : float
                The frequency of the signal.
            filtered_signal : np.ndarray
                Signal filtered using Butterworth filter.

        Notes
        -----
        1. Frequency is defined as:

        .. math::

            f = \frac{1}{T}

           `period` corresponds to :math:`T`

        2. `order` is passed as the `N` parameter when calling
           `scipy.signal.butter`.
        3. According to [1]_, BIC should be used instead of AIC for this
           computation. The brief rationale is .......

        References
        ----------
        .. [1] Author, "Why BIC is better than AIC for AR model", Statistics,
               vol. 1, pp. 1-15, 1996.

        """
        # We use Butterworth filter from scipy to perform some calculation.
        # Note that parameter `N` is passed using keys, taking the value of the
        # `order` input parameter
        filtered_signal = scipy.signal.butter(N=order, ...)

        # Here we calculate a return value using a function variable. Note that
        # this variable is named in the "Returns" section
        frequency = 1 / period
        return frequency, filtered_signal



Class docstring
---------------

Class docstrings follow function docstring format. Here is an example.

.. code-block:: python

    class MyClass(object):  # Classes use CamelCase notation
        """
        One line description of class.

        Long description of class, may span several lines. Possible sections are
        the same as for a function doc, with additional "Attributes" and "Methods"
        after "Parameters" (cf. numpydoc guide). Do not put a blank line after
        section headers, do put a blank line at the end of a long docstring.

        When explaining the algorithm, you can use mathematical notation, e.g.:

        .. math::

            E = m c^2

        To insert an equation use `.. math::` and surround the whole expression in
        blank lines. To use math notation in-line, write :math:`E` corresponds to
        energy and :math:`m` to mass. Embed expressions after `:math:` in
        backticks, e.g. :math:`x^2 + y^2 = z^2`.

        To refer to a paper in which formula is described, use the expression
        "see [1]_" - it will become an interactive link on the readthedocs website.
        The underscore after the closing bracket is mandatory for the link to
        work.

        To refer to a note in the "Notes" section, simply write "see Notes [1]".

        Variable, module, function, and class names should be written
        between single back-ticks (`kernels.AlphaKernel`), NOT *bold*.

        For common modules such as Numpy and Quantities, use the notation
        according to the import statement. For example:
        "this function uses `np.diff`", not "uses `numpy.diff`".

        Prefixes for common packages in Elephant are the following:

        1. Neo objects = neo (e.g. `neo.SpikeTrain`)
        2. Numpy = np (e.g. `np.ndarray`)
        3. Quantities = pq (e.g. `pq.Quantity`)

        For other objects, list the full path to the object (e.g., for the
        BinnedSpikeTrain, this would be `elephant.conversion.BinnedSpikeTrain`)

        For None and NaNs, do not use backticks. NaN is referred as np.NaN (i.e.,
        with the Numpy prefix "np").

        Use backticks also when referring to arguments of a function (e.g., `x` or
        `y`), and :attr:`attribute_name` when referring to attributes of a class
        object in docstrings of this class.

        To refer to attributes of other objects, write
        `other_object.relevant_attribute` (e.g. `neo.SpikeTrain.t_stop`).

        When mentioning a function from other module, type `other_module.function`
        (without parentheses after the function name; e.g., `scipy.signal.butter`).

        If you refer values to True/False/None, do not use backticks, unless an
        emphasis is needed. In this case, write `True` and not bold, like **True**.

        Parameters
        ----------
        <!-- List the arguments of the constructor (__init__) here!
        Arguments must come in the same order as in the constructor or function -->
        parameter : int or float
            Description of parameter `parameter`. Enclose variables in single
            backticks. The colon must be preceded by a space.
        no_type_parameter
            Colon omitted if the type is absent.
        x : float
            The X coordinate.
        y : float
            The Y coordinate.
            Default: 1.0.  <!-- not "Default is 1.0." (it is just a convention) -->
        z : float or int or pq.Quantity
            This is Z coordinate.
            If it can take multiple types, separate them by "or", do not use commas
            (numpy style).
            If different actions will happen depending on the type of `z`, explain
            it briefly here, not in the main text of the function/class docstring.
        s : {'valid', 'full', 'other'}
            This is the way to describe a list of possible argument values, if the
            list is discrete and predefined (typically concerns strings).
            If 'valid', the function performs some action.
            If 'full', the function performs another action.
            If 'other', the function will ignore the value defined in `z`.
            Default: 'valid'.
        spiketrains : neo.SpikeTrain or list of neo.SpikeTrain or np.ndarray
            When the parameter can be a container (such as list or tuple), you can
            specify the type of elements using "of". But use the Python type name
            (do not add "s" to make it plural; e.g., do not write
            "list of neo.SpikeTrains" or "list of neo.SpikeTrain objects").
        counts_matrix : (N, M) np.ndarray
            This is the way to indicate dimensionality of the required array
            (i.e.,if the function only works with 2D-arrays). `N` corresponds to
            the number of rows and `M` to the number of columns. Refer to the same
            `N` and `M` to describe the dimensions of the returned values when
            they are determined by the dimensions of the parameter.
        is_true : bool
            True, if 1.
            False, if 0.
            Default: True.
        other_parameter : int
            Some value.
            If value is None and the function takes some specific action (e.g.,
            calculate some value based on the other inputs), describe here.
            Default: None.

        Attributes
        ----------
        <!-- Here list the attributes of class object which are not simply copies
        of the constructor parameters. Property decorators (@property) are also
        considered attributes -->
        a : list
            This is calculated based on `x` and `y`.
        b : int
            This is calculated on the way, during some operations.

        Methods
        -------
        <!--  Here list the most important/useful class methods (not all the
        methods) -->

        Returns
        -------
        <!-- This section is rarely used in class docstrings, but often in
        function docs. Follow the general recommendation of numpydoc.
        If there is more than one returned value, use variable names for the
        returned value, like `error_matrix` below. -->
        error_matrix : np.ndarray
            A matrix is stored in a variable called `error_matrix`, containing
            errors estimated from some calculations. The function "return"
            statement then returns the variable (e.g. "return error_matrix").
            Format is the same as for any parameter in section "Parameters".
            Use meaningful names, not general names such as `output` or `result`.
        list
            The returned object is created on the fly and is never assigned to
            a variable (e.g. "return [1, 2, 3]"). Simply name the type and
            describe the content. This should be used only if the function returns
            a single value.
        dict
            key_1 : type
                Description of key_1, formatted the same as in "Parameters".
            key_2 : type
                Description of key_2
        particular_matrix : (N, N, M) np.ndarray
            The dimensionality of this array depends on the dimensionality of
            `counts_matrix` input parameter. Note that `N` and `M` are used since
            these were the names of the dimensions of `counts_matrix` in the
            "Parameters" section.
        list_variable : list of np.ndarray
            Returns a list of numpy arrays.
        signal : int
            Description of `signal`.

        Raises
        ------
        <!-- List the errors explicitly raised by the constructor (raise
        statements), even if they are in fact raised by other Elephant functions
        called inside the constructor. Enumerate them in alphabetical order. -->
        TypeError
            If `x` is an `int` or None.
            If `y` is not a `float`.
        ValueError
            If this and that happens.

        Warns
        -----
        <!-- Here apply the same rules as for "Raises". -->
        UserWarning
            If something may be wrong but does not prevent execution of the code.
            The default warning type is UserWarning.

        Warning
        -------
        <!-- Here write a message to the users to warn them about something
        important.
        Do not enumerate Warnings in this section! -->

        See Also
        --------
        <!-- Here refer to relevant functions (also from other modules). Follow
        numpydoc recommendations.
        If the function name is not self-explanatory, you can add a brief
        explanation using a colon separated by space.
        This items will be placed as links to the documentation of the function
        referred.
        -->
        statistics.isi
        scipy.signal.butter : Butterworth filter

        Notes
        -----
        <!-- Here you can add some additional explanations etc. If you have several
        short notes (at least two), use a list -->
        1. First remark.
        2. Second much longer remark, which will span several lines. To refer to a
           note in other parts of the docstring, use a phrase like "See Notes [2]".
           To make sure that the list displays correctly, keep the indentation to
           match the first word after the point (as in this text).
        3. If you want to explain why the default value of an argument is
           something particular, you can give a more elaborate explanation here.
        4. If the function has an alias (see the last function in this file), the
           information about it should be in this section in the form:
           Alias: bla.
           Aliases should be avoided.
        5. Information about validation should be here, and insert bibliographic
           citation in the "References". Also specify in parenthesis the unit test
           that implements the validation. Example:
           "This function reproduces the paper Riehle et al., 1997 [2]_.
           (`UETestCase.test_Riehle_et_al_97_UE`)."
        6. Do not create new section names, because they will not be displayed.
           Place the relevant information here instead.
        7. This is an optional section that provides additional information about
           the code, possibly including a discussion of the algorithm. This
           section may include mathematical equations, written in LaTeX format.
           Inline: :math:`x^2`. An equation:

           .. math::

           x(n) * y(n) \Leftrightarrow X(e^{j\omega } )Y(e^{j\omega } )

        8. Python may complain about backslashes in math notation in docstrings.
           To prevent the complains, precede the whole docstring with "r" (raw
           string).
        9. Images are allowed, but should not be central to the explanation;
           users viewing the docstring as text must be able to comprehend its
           meaning without resorting to an image viewer. These additional
           illustrations are included using:

            .. image:: filename

        References
        ----------
        .. [1] Smith J., "Very catchy title," Elephant 1.0.0, 2020. The ".." in
               front makes the ref referencable in other parts of the docstring.
               The indentation should match the level of the first word AFTER the
               number (in this case "Smith").

        Examples
        --------
        <!-- If applicable, provide some brief description of the example, then
        leave a blank line.
        If the second example uses an import that was already used in the first
        example, do not write the import again.
        Examples should be very brief, and should avoid plotting. If plotting
        is really needed, use simple matplotlib plots, that take only few lines.
        More complex examples, that require lots of plotting routines (e.g.,
        similar to Jupyter notebooks), should be placed as tutorials, with links
        in the docstring. Examples should not load any data, but only use easy
        generated data.
        Finally, avoid using abbreviations in examples, such as
        "import elephant.conversion as conv" -->

        >>> import neo
        >>> import numpy as np
        >>> import quantities as pq
        ...
        ... # This is a way to make a blank line within the example code.
        >>> st = neo.SpikeTrain([0, 1, 2, 3] * pq.ms, t_start=0 * pq.ms,
        ...                     t_stop=10 * pq.ms, sampling_rate=1 * pq.Hz)
        ... # Use "..." also as a continuation line.
        >>> print(st)
        SpikeTrain

        Here provide a brief description of a second example. Separate examples
        with a blank line even if you do not add any description.

        >>> import what_you_need
        ...
        >>> st2 = neo.SpikeTrain([5, 6, 7, 8] * pq.ms, t_start=0 * pq.ms,
        ...                      t_stop=10 * pq.ms, sampling_rate=1 * pq.Hz)
        >>> sth = what_you_need.function(st2)
        >>> sth_else = what_you_need.interesting_function(sth)

        """

        def __init__(self, parameter):
            """
            Constructor
            (actual documentation is in class documentation, see above!)
            """
            self.parameter = parameter
            self.function_a()  # creates new attribute of self 'a'

        def function_a(self, parameter, no_type_parameter, spiketrains,
                       is_true=True, string_parameter='C', other_parameter=None):
            """
            One-line short description of the function.

            Long description of the function. Details of what the function is doing
            and how it is doing it. Used to clarify functionality, not to discuss
            implementation detail or background theory, which should rather be
            explored in the "Notes" section below. You may refer to the parameters
            and the function name, but detailed parameter descriptions still
            belong in the "Parameters" section.

            Parameters
            ----------
            <!-- See class docstring above -->

            Returns
            -------
            <!-- See class docstring above -->

            Raises
            ------
            <!-- See class docstring above.
            List only exceptions explicitly raised by the function -->

            Warns
            -----
            <!-- See class docstring above. -->

            See Also
            --------
            <!-- See class docstring above  -->

            Notes
            -----
            <!-- See class docstring above -->

            References
            ----------
            <!-- See class docstring above -->

            Examples
            --------
            <!-- See class docstring above -->

            """

            # Variables use underscore notation
            dummy_variable = 1
            a = 56  # This mini comment uses two spaces after the code!

            # Textual strings use double quotes
            error = "An error occurred. Please fix it!"
            # Textual strings are usually meant to be printed, returned etc.

            # Non-textual strings use single quotes
            default_character = 'a'
            # Non textual strings are single characters, dictionary keys and other
            # strings not meant to be returned or printed.

            # Normal comments are proceeded by a single space, and begin with a
            # capital letter
            dummy_variable += 1

            # Longer comments can have several sentences. These should end with a
            # period. Just as in this example.
            dummy_variable += 1

        # Class functions need only 1 blank line.
        # This function is deprecated. Add a warning!
        def function_b(self, **kwargs):
            """
            This is a function that does b.

            .. deprecated:: 0.4
              `function_b` will be removed in elephant 1.0, it is replaced by
              `function_c` because the latter works also with Numpy Ver. 1.6.

            Parameters
            ----------
            kwargs : dict
                kwarg1 : type
                    Same style as docstring of class `MyClass`.
                kwarg2 : type
                    Same style as docstring of class `MyClass`.

            """
            pass
