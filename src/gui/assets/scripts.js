let isMouseDown = false;

document.addEventListener('mousedown', function(event) {
    if ( event.key ) isMouseDown = true;
}, true);

document.addEventListener('mouseup', function(event) {
    if ( event.key ) isMouseDown = false;
}, true);