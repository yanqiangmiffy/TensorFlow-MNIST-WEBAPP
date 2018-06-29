"use strict";

function _classCallCheck(t, e) {
    if (!(t instanceof e)) throw new TypeError("Cannot call a class as a function")
}

var _createClass = function () {
    function t(t, e) {
        for (var i = 0; i < e.length; i++) {
            var n = e[i];
            n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(t, n.key, n)
        }
    }

    return function (e, i, n) {
        return i && t(e.prototype, i), n && t(e, n), e
    }
}(), Main = function () {
    function t() {
        _classCallCheck(this, t), this.canvas = document.getElementById("main"), this.input = document.getElementById("input"), this.canvas.width = 449, this.canvas.height = 449, this.ctx = this.canvas.getContext("2d"), this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this)), this.canvas.addEventListener("mouseup", this.onMouseUp.bind(this)), this.canvas.addEventListener("mousemove", this.onMouseMove.bind(this)), this.initialize()
    }

    return _createClass(t, [{
        key: "initialize", value: function () {
            this.ctx.fillStyle = "#FFFFFF", this.ctx.fillRect(0, 0, 449, 449), this.ctx.lineWidth = 1, this.ctx.strokeRect(0, 0, 449, 449), this.ctx.lineWidth = .05;
            for (var t = 0; t < 27; t++) this.ctx.beginPath(), this.ctx.moveTo(16 * (t + 1), 0), this.ctx.lineTo(16 * (t + 1), 449), this.ctx.closePath(), this.ctx.stroke(), this.ctx.beginPath(), this.ctx.moveTo(0, 16 * (t + 1)), this.ctx.lineTo(449, 16 * (t + 1)), this.ctx.closePath(), this.ctx.stroke();
            this.drawInput(), $("#output td").text("").removeClass("bg-success")
        }
    }, {
        key: "onMouseDown", value: function (t) {
            this.canvas.style.cursor = "default", this.drawing = !0, this.prev = this.getPosition(t.clientX, t.clientY)
        }
    }, {
        key: "onMouseUp", value: function () {
            this.drawing = !1, this.drawInput()
        }
    }, {
        key: "onMouseMove", value: function (t) {
            if (this.drawing) {
                var e = this.getPosition(t.clientX, t.clientY);
                this.ctx.lineWidth = 16, this.ctx.lineCap = "round", this.ctx.beginPath(), this.ctx.moveTo(this.prev.x, this.prev.y), this.ctx.lineTo(e.x, e.y), this.ctx.stroke(), this.ctx.closePath(), this.prev = e
            }
        }
    }, {
        key: "getPosition", value: function (t, e) {
            var i = this.canvas.getBoundingClientRect();
            return {x: t - i.left, y: e - i.top}
        }
    }, {
        key: "drawInput", value: function () {
            var t = this.input.getContext("2d"), e = new Image;
            e.onload = function () {
                var i = [], n = document.createElement("canvas").getContext("2d");
                n.drawImage(e, 0, 0, e.width, e.height, 0, 0, 28, 28);
                for (var s = n.getImageData(0, 0, 28, 28).data, a = 0; a < 28; a++) for (var o = 0; o < 28; o++) {
                    var c = 4 * (28 * a + o);
                    i[28 * a + o] = (s[c + 0] + s[c + 1] + s[c + 2]) / 3, t.fillStyle = "rgb(" + [s[c + 0], s[c + 1], s[c + 2]].join(",") + ")", t.fillRect(5 * o, 5 * a, 5, 5)
                }
                255 !== Math.min.apply(Math, i) && $.ajax({
                    url: "/api/mnist",
                    method: "POST",
                    contentType: "application/json",
                    data: JSON.stringify(i),
                    success: function (t) {
                        for (var e = 0; e < 2; e++) {
                            for (var i = 0, n = 0, s = 0; s < 10; s++) {
                                var a = Math.round(1e3 * t.results[e][s]);
                                a > i && (i = a, n = s);
                                for (var o = String(a).length, c = 0; c < 3 - o; c++) a = "0" + a;
                                var r = "0." + a;
                                a > 999 && (r = "1.000"), $("#output tr").eq(s + 1).find("td").eq(e).text(r)
                            }
                            for (var h = 0; h < 10; h++) h === n ? $("#output tr").eq(h + 1).find("td").eq(e).addClass("bg-success") : $("#output tr").eq(h + 1).find("td").eq(e).removeClass("bg-success")
                        }
                    }
                })
            }, e.src = this.canvas.toDataURL()
        }
    }]), t
}();
$(function () {
    var t = new Main;
    $("#clear").click(function () {
        t.initialize()
    })
});