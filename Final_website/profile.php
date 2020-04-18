<?php
include "php/config.php";
session_start();
$user= $_SESSION["loggedin_user"];

$sql="SELECT * from patients WHERE email_id ='".$_SESSION["loggedin_user"]."'";
$result=$conn->query($sql);
while ($row=mysqli_fetch_assoc($result))
{
    $_SESSION["id"]=$row['id'];
    $_SESSION["name"] = $row['name'];
    $_SESSION["age"] = $row['age'];
    $_SESSION["dob"] = $row['date of birth'];
    $_SESSION["sex"] = $row['sex'];
    $_SESSION["blood_group"] = $row['blood_group'];
    $_SESSION["contact"] = $row['contact'];
    $_SESSION["address"] = $row['address'];
    $_SESSION["medical_history"] = $row['medical_history'];
    $_SESSION["email"] = $row['email_id'];
    
}

echo $_SESSION["id"];
echo $_SESSION["name"];
echo $_SESSION["age"];
echo $_SESSION["dob"];
echo $_SESSION["sex"];
echo $_SESSION["blood_group"];
echo $_SESSION["contact"];
echo $_SESSION["email"];
echo $_SESSION["medical_history"];




?>