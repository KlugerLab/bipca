{% if objtype == 'method' %}
:orphan:
{% endif %}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == 'method' %}

{% endif %}

.. auto{{ objtype }}:: {{ fullname | replace("numpy.", "numpy::") }}

{# In the fullname (e.g. `numpy.ma.MaskedArray.methodname`), the module name
is ambiguous. Using a `::` separator (e.g. `numpy::ma.MaskedArray.methodname`)
specifies `numpy` as the module name. #}
