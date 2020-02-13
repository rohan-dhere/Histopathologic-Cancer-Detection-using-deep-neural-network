<?php
$dbservername = "localhost";
$dbusername = "root";
$dbpassword = "123456";
$dbname="cancer_detection";
$conn = mysqli_connect($dbservername, $dbusername, $dbpassword,$dbname);


if ( !$conn ) {
    echo ( 'Did not connect: ' . mysqli_connect_error() ); 
}

?>